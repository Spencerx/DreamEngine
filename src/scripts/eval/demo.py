# Build a gradio demo for text-image interleaved image generation
# Input: two images and their corresponding captions, plus a text instruction
# For example:
# Input:
# Image 1: A dog image
# Image 1 caption: A dog
# Image 2: A cat image
# Image 2 caption: A cat
# Text instruction: Combine the two animals into one animal
# Output: A synthesized animal image

# The model also accepts some parameters:
# cfg_scale: A scalar to control the quality of generated image
# size: From 512x512 to 1024x1024
# num_steps: 28
# seed: A scalar to control the randomness of generated image

# Temporarily hardcode the generation function, input is image1, image2, caption1, caption2, text_prompt, cfg_scale, size, num_steps, seed
# Output is the generated image

# Consolidated imports
import os
import json
import torch
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from diffusers.models.transformers.transformer_sd3 import (
    SD3Transformer2DModel,
    QwenVLSD3_DirectMap_Transformer2DModel as QwenVLSD3Transformer2DModel
)
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig

# set CUDA_VISIBLE_DEVICES to 0
# Constants
MODEL_PATH = "/home/data/cl/models"
QWEN_PATH = f"{MODEL_PATH}/Qwen2-VL-2B-Instruct"
SD3_PATH = f"{MODEL_PATH}/stable-diffusion-3.5-large"
DreamEngine_CKPT_DIR="/home/data/cl/Dream-Engine/ckpt/any-regen-objectmodel60k_lmmditlora32"

# Model initialization
qwenvl2_model = Qwen2VLForConditionalGeneration.from_pretrained(QWEN_PATH)
sd3_model = SD3Transformer2DModel.from_pretrained(f"{SD3_PATH}/transformer")

# LoRA configurations
lmm_lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    init_lora_weights="gaussian",
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ]
)

transformer_lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    init_lora_weights="gaussian",
    target_modules=[
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
        "attn.to_add_out", "attn.to_k", "attn.to_out.0",
        "attn.to_q", "attn.to_v",
    ]
)

# Apply LoRA configurations
qwenvl2_model.add_adapter(lmm_lora_config)
sd3_model.add_adapter(transformer_lora_config)

# Initialize other components
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    SD3_PATH, subfolder="scheduler"
)
processor = AutoProcessor.from_pretrained(QWEN_PATH, max_pixels=512*28*28)
vae = AutoencoderKL.from_pretrained(
    SD3_PATH,
    subfolder="vae"
).to("cuda", dtype=torch.bfloat16)

def load_sharded_model(config_path, index_path, bin_files_folder, device='cpu',dtype=torch.bfloat16):
    """
    Loads a sharded Hugging Face model from multiple binary files.

    Args:
        config_path (str): Path to the model configuration JSON file.
        index_path (str): Path to the model index JSON file.
        bin_files_folder (str): Directory containing the binary model files.
        device (str): Device to load the model onto ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded model with weights.
    """
    # Step 1: Load the Model Configuration
    print("Loading model configuration...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize the model using the configuration
    print("Initializing the model based on the configuration...")
    model = QwenVLSD3Transformer2DModel(qwenvl2_model, sd3_model)
    
    # Step 2: Load the Model Index
    print("Loading model index file...")
    with open(index_path, 'r') as f:
        index = json.load(f)
    
    weight_map = index.get('weight_map', {})
    if not weight_map:
        raise ValueError("The index file does not contain a 'weight_map' key.")
    
    # Step 3: Organize Weights by Binary File
    print("Organizing weights by their respective binary files...")
    bins = {}
    for weight_name, bin_file in weight_map.items():
        bins.setdefault(bin_file, []).append(weight_name)
    
    # Initialize an empty state dictionary
    state_dict = {}
    
    # Step 4: Load Each Binary File and Extract Relevant Weights
    for bin_file, weight_names in bins.items():
        bin_path = os.path.join(bin_files_folder, bin_file)
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(f"Binary file not found: {bin_path}")
        
        print(f"Loading binary file: {bin_path}")
        bin_state = torch.load(bin_path, map_location="cpu")
        
        # Determine how the weights are stored in the binary file
        # Common scenarios:
        # a) The entire state_dict is stored directly
        # b) The state_dict is nested under a key like 'state_dict'

        if isinstance(bin_state, dict):
            if 'state_dict' in bin_state:
                partial_state = bin_state['state_dict']
            else:
                partial_state = bin_state

            # Extract only the weights relevant to this bin file
            for weight_name in weight_names:
                if weight_name in partial_state:
                    state_dict[weight_name] = partial_state[weight_name]
                else:
                    print(f"Warning: '{weight_name}' not found in '{bin_file}'.")
        else:
            raise ValueError(f"Unexpected format in binary file: {bin_file}")

    # Step 5: Load the Merged State Dictionary into the Model
    print("Loading weights into the model...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print("Warning: The following keys are missing in the state dictionary:")
        for key in missing_keys:
            print(f"  - {key}")
    if unexpected_keys:
        print("Warning: The following keys are unexpected in the state dictionary:")
        for key in unexpected_keys:
            print(f"  - {key}")
    
    # Transfer the model to the specified device
    print(f"Transferring the model to {device.upper()}...")
    model.to(dtype=dtype).to(device)  # First change dtype, then device
    model.eval()  # Set the model to evaluation mode

    print("Model loaded successfully.")
    return model





model = load_sharded_model(
    config_path=DreamEngine_CKPT_DIR+"/transformer/config.json",
    index_path=DreamEngine_CKPT_DIR+"/transformer/diffusion_pytorch_model.bin.index.json",
    bin_files_folder=DreamEngine_CKPT_DIR+"/transformer",
    device='cuda',
    dtype=torch.bfloat16
)

from diffusers.pipelines.stable_diffusion_3.pipeline_qwen_vl_stable_diffusion_3 import QwenVLStableDiffusion3Pipeline

pipeline = QwenVLStableDiffusion3Pipeline(
    model,
    processor,
    noise_scheduler,
    vae
)

from PIL import Image
from torchvision import transforms


obj_transform = transforms.Compose(
    [
        transforms.Resize(336, interpolation=transforms.InterpolationMode.BILINEAR),
    ]
)




def generate_image(image1, image2, caption1, caption2, text_prompt, cfg_scale, size, num_steps, seed):
    """Generate an image based on input parameters."""
    # Input validation
    if any(x is None for x in [image1, image2]):
        raise gr.Error("Please upload two input images")
    if not all([caption1, caption2, text_prompt]):
        raise gr.Error("Please provide all text inputs")
    if not all(cap in text_prompt for cap in [caption1, caption2]):
        raise gr.Error("Text prompt must include descriptions of both images")
    

    torch.manual_seed(seed)

    segments = [caption1,obj_transform(image1)],[caption2,obj_transform(image2)]

    output = pipeline.cfg_predict(prompt=text_prompt,segments=segments,num_inference_steps=num_steps,num_images_per_prompt=1,width=size,height=size, guidance_scale=cfg_scale, max_sequence_length=334)

    return output[0][0] 

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("""
    # Dream Engine Text-Guided Object Fusion Demo

    This demo allows you to combine elements from two images based on your text instructions.
    
    **Project Links:**
    - [GitHub Repository](https://github.com/chenllliang/DreamEngine)
    - [Research Paper](https://arxiv.org/abs/2502.20172)

    **Note:**
    - Average generation time: ~30 seconds
    - The model may sometimes produce unrealistic or unexpected results
    - **Warning:** Generated content may sometimes causing uncomfortable feelings, there is no NSFW filter.
    
    **Instructions:**
    1. Upload two images
    2. Add descriptive tags for each image
    3. Write an instruction that includes both image tags
    4. Adjust generation parameters as needed
                
    *Check the example inputs in the bottom*
    """)

    
    
    with gr.Row():
        with gr.Column():
            image1_input = gr.Image(label="Image 1",type="pil") #pil image
            caption1_input = gr.Textbox(label="Image 1 Tag")
        with gr.Column():
            image2_input = gr.Image(label="Image 2",type="pil")
            caption2_input = gr.Textbox(label="Image 2 Tag")
    
    text_prompt = gr.Textbox(label="Text Instruction (Must include both image tags)", placeholder="Please enter synthesis instruction...")
    
    with gr.Row():
        cfg_scale = gr.Slider(minimum=2.0, maximum=20.0, value=3.5, step=0.5, label="CFG Scale")
        size = gr.Slider(minimum=512, maximum=1024, value=768, step=64, label="Image Size")
        num_steps = gr.Slider(minimum=1, maximum=100, value=28, step=1, label="Steps")
        seed = gr.Slider(minimum=0, maximum=1000000, value=136147, step=1, label="Random Seed")
    
    generate_btn = gr.Button("Generate Image")
    output_image = gr.Image(label="Generated Result")
    
    generate_btn.click(
        fn=generate_image,
        inputs=[
            image1_input, image2_input,
            caption1_input, caption2_input,
            text_prompt,
            cfg_scale, size, num_steps, seed
        ],
        outputs=output_image
    )
    
    # Add example inputs
    gr.Examples(
        examples=[
            [
                Image.open("/home/data/cl/Dream-Engine/test_images/test_image/1665_Girl_with_a_Pearl_Earring.jpg"),Image.open("/home/data/cl/Dream-Engine/test_images/orange_cat.png"), 
                "girl","cat",
                "a girl with a cat",
                3.5, 768, 28, 136147
            ],
            [
                Image.open("/home/data/cl/Dream-Engine/test_images/orange_cat.png"), Image.open("/home/data/cl/Dream-Engine/test_images/forest.png"),
                "cat","forest",
                "a cat sitting in the forest",
                3.5, 1024, 28, 339582
            ]

        ],
        inputs=[
            image1_input, image2_input,
            caption1_input, caption2_input,
            text_prompt,
            cfg_scale, size, num_steps, seed
        ]
    )
    

if __name__ == "__main__":
    demo.launch(share=True)





