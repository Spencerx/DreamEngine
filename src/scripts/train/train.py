import argparse
import os

from diffusers.pipelines.stable_diffusion_3.pipeline_qwen_vl_stable_diffusion_3 import QwenVLStableDiffusion3Pipeline
from diffusers.models.transformers.transformer_sd3 import QwenVLSD3Transformer2DModel, SD3Transformer2DModel, QwenVLSD3_DirectMap_Transformer2DModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import requests

from peft import LoraConfig, set_peft_model_state_dict
from torchvision.datasets import ImageFolder


import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module


import wandb
from datasets import load_dataset


logger = get_logger(__name__)

import os
import json

def get_captions_with_random_objects(segments, image, max_obj_num=3, max_size=0.5):
    caption_object_list = []
    # Calculate the max allowed object area based on max_size
    max_object_area = max_size * image.width * image.height

    for segment in segments:
        caption, boxes = segment

        caption_object_list.append(caption)

        if boxes:
            for box in boxes:
                x1, y1, x2, y2 = box
                # Scale the coordinates according to the image size
                x1, x2 = x1 * image.width / 1000, x2 * image.width / 1000
                y1, y2 = y1 * image.height / 1000, y2 * image.height / 1000
                width, height = x2 - x1, y2 - y1
                # Calculate the area of the object
                object_area = width * height
                # Check if the object's area is within the allowed maximum size
                if object_area <= max_object_area:
                    # Crop the object from the image
                    object_image = image.crop((x1, y1, x2, y2))
                    # Append (name, object_image) to the list
                    caption_object_list.append(object_image)

    # If number of objects exceeds max_obj_num, randomly select max_obj_num objects from all objects, do not change the captions, only filter the object image

    # get the index of object from caption_object_list
    object_index = [i for i in range(len(caption_object_list)) if type(caption_object_list[i]) == Image.Image]
    caption_index = [i for i in range(len(caption_object_list)) if type(caption_object_list[i]) == str]

    # filter the object image
    if len(object_index) > max_obj_num:
        object_index = random.sample(object_index, max_obj_num)

    # remove the object not in object_index
    keep_index = sorted(object_index + caption_index)
    caption_object_list = [caption_object_list[i] for i in keep_index]

    #  example:   [' Two metal chairs',
    #  <PIL.Image.Image image mode=RGB size=587x811>,
    #  ' with',
    #  ' grey cushions',
    #  <PIL.Image.Image image mode=RGB size=404x146>,
    #  ',',
    #  ' turquoise pillows',
    #  <PIL.Image.Image image mode=RGB size=332x193>,
    #  <PIL.Image.Image image mode=RGB size=499x255>,
    #  ', and',
    #  ' a table',
    #  <PIL.Image.Image image mode=RGB size=837x309>,
    #  ' with flowers on outdoor patio']

    return caption_object_list


def get_random_objects_with_name(segments, image, max_obj_num=3, max_size=0.5):
    # A list to store tuples of (name, object_image)
    objects_with_name = []

    # Calculate the max allowed object area based on max_size
    max_object_area = max_size * image.width * image.height

    for segment in segments:
        caption, boxes = segment
        if boxes:
            for box in boxes:
                x1, y1, x2, y2 = box
                # Scale the coordinates according to the image size
                x1, x2 = x1 * image.width / 1000, x2 * image.width / 1000
                y1, y2 = y1 * image.height / 1000, y2 * image.height / 1000
                width, height = x2 - x1, y2 - y1
                # Calculate the area of the object
                object_area = width * height
                # Check if the object's area is within the allowed maximum size
                if object_area <= max_object_area:
                    # Crop the object from the image
                    object_image = image.crop((x1, y1, x2, y2))
                    # Append (name, object_image) to the list
                    objects_with_name.append((caption.strip(), object_image))

    # If number of objects exceeds max_obj_num, randomly select max_obj_num objects
    if len(objects_with_name) > max_obj_num:
        objects_with_name = random.sample(objects_with_name, max_obj_num)

    return objects_with_name



import torch
from torchvision import transforms
from torchvision.transforms import functional as F

class RoundTo16:
    def __init__(self):
        pass

    def __call__(self, img):
        # Get original dimensions
        width, height = img.size

        # Round dimensions to nearest multiple of 16
        new_width = round(width / 16) * 16
        new_height = round(height / 16) * 16

        # Resize image to new dimensions
        return F.resize(img, (new_height, new_width), interpolation=F.InterpolationMode.BILINEAR)



class InterleavedDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        size=512,
        center_crop=False,
        cache_dir=None,
    ):
        self.size = size
        self.center_crop = center_crop


        print("loading dataset")
      
        # Load the dataset

        self.image_column = dataset_config['image_column']
        self.caption_column = dataset_config['text_column']
        

        self.target_image_column = dataset_config.get("target_image_column", None)

        self.segment_column = dataset_config.get("segment_column", None)





        self.is_jsonl_dataset = dataset_config['type']=='jsonl'

        self.task = dataset_config['task']

        if dataset_config['type']=='image_folder':
            self.dataset = ImageFolder(dataset_config['path'])
        elif dataset_config['type']=='jsonl':
            self.dataset = load_dataset("json",data_files=dataset_config['path'])['train']
        elif dataset_config['type']=='hf':
            self.dataset = load_dataset(
                dataset_config['path'],
                cache_dir=cache_dir,
            )['train']

        elif dataset_config['type']=='hf-ultraedit':
            self.dataset = load_dataset("parquet", data_files=[os.path.join("./datasets/UltraEdit/data",i) for i in os.listdir("./datasets/UltraEdit/data")[:1000]],cache_dir=cache_dir)['train']
        
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_config['type']}")

        
        
        print("loading successful")
        print("center_crop",center_crop)

        # Define the train transformations
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.crop_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )

        self.object_image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, max_size=600),
                RoundTo16(),
            ]
        )

        # Define the transform pipeline
        self.object_image_transforms_pixel = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR, max_size=600),
                RoundTo16(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        print("init successful")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        # Load and process image
        try:
            if self.is_jsonl_dataset:
                image_url = self.dataset[index][self.image_column]
                if image_url.startswith("http"):
                    image = Image.open(requests.get(image_url, stream=True).raw)
                else:
                    image = Image.open(image_url)
                
                if self.target_image_column:
                    target_image_url = self.dataset[index][self.target_image_column]
                    if target_image_url.startswith("http"):
                        target_image = Image.open(requests.get(target_image_url, stream=True).raw)
                    else:
                        target_image = Image.open(target_image_url)
                
                if self.segment_column:
                    segments = eval(self.dataset[index][self.segment_column]) 
            
            else: 
                image = self.dataset[index][self.image_column]

                if self.target_image_column:
                    target_image = self.dataset[index][self.target_image_column]
                    


            
            
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")

        
        
        
        except Exception as e:
            # random pick an image , re get the item
            print(e)
            new_index = random.randint(0,len(self.dataset)-1)
            return self.__getitem__(new_index)
        
        

        
        
        example["tensor_image"] = self.image_transforms(image)

        example["target_image"] = self.image_transforms(target_image) if self.target_image_column else None

        
        example["target_image"] = self.object_image_transforms_pixel(image) if self.segment_column else None


        if self.task == "OBJ2I_V2":
            example["obj_name_image"] = get_captions_with_random_objects(segments, self.object_image_transforms(image))
        elif self.task == "OBJ2I":
            example["obj_name_image"] = get_random_objects_with_name(segments, self.object_image_transforms(image))
        else:
            example["obj_name_image"] = None 

        

        

        example["pil_image"] = self.crop_transforms(image)

        
        if self.caption_column:
            example["prompt"] = str(self.dataset[index][self.caption_column])
        else:
            example["prompt"] = None
        

        example["task"] = self.task

        
        
        
        return example


class ConcatDataset(Dataset):
    def __init__(self, datasets, dataset_lengths):
        """
        Initializes the ConcatDataset with a list of datasets and respective lengths.
        
        :param datasets: List of datasets to concatenate. Each should be an instance of a subclass of Dataset.
        :param dataset_lengths: List of integers specifying how many samples to take from each dataset.
        """
        assert len(datasets) == len(dataset_lengths), "Length of datasets and dataset_lengths must match"
        self.datasets = datasets
        self.dataset_lengths = dataset_lengths
        self.cumulative_sizes = self._compute_cumulative_sizes()

    def _compute_cumulative_sizes(self):
        cumulative_sizes = [sum(self.dataset_lengths[:i+1]) for i in range(len(self.dataset_lengths))]
        return cumulative_sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        dataset_idx = self._find_dataset(index)
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        
        # Ensure sample index is within the desired length for the respective dataset
        if sample_idx >= self.dataset_lengths[dataset_idx]:
            raise IndexError("Sample index out of range for dataset length constraint.")
        
        return self.datasets[dataset_idx][sample_idx]

    def _find_dataset(self, index):
        for i, size in enumerate(self.cumulative_sizes):
            if index < size:
                return i
        raise ValueError("Index out of range")


def collate_fn(examples):
    pixel_values = [example["tensor_image"] for example in examples]
    prompts = [example["prompt"] for example in examples]
    images = [example["pil_image"] for example in examples]
    tasks = [example["task"] for example in examples]

    obj_name_images = [example["obj_name_image"] for example in examples]



    targets = [example["target_image"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    targets_pixel_values = torch.stack(targets) if targets[0] is not None else None
    targets_pixel_values = targets_pixel_values.to(memory_format=torch.contiguous_format).float() if targets_pixel_values is not None else None


    batch = {"tensor_image": pixel_values, "prompt": prompts, "pil_image": images, "task": tasks, "target_image": targets_pixel_values, "obj_name_image":obj_name_images}
    return batch

def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args_list,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
    )
    # pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()

    with autocast_ctx:
        if "prompt" in pipeline_args_list[0]:
            images = [pipeline.cfg_predict(**pipeline_args, generator=generator, guidance_scale=3.5).images[0] for pipeline_args in pipeline_args_list]
        else:
            images = [pipeline(**pipeline_args, generator=generator).images[0] for pipeline_args in pipeline_args_list]
    
    
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"

        if "prompt" in pipeline_args_list[0]:
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {pipeline_args_list[i]['prompt']}") for i, image in enumerate(images)
                    ]
                }
            )
        else:
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption="Reconstruction") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()

    return images

def load_sharded_model(config_path, index_path, bin_files_folder, qwenvl2_model, sd3_model, device='cpu'):
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
    model = QwenVLSD3_DirectMap_Transformer2DModel(qwenvl2_model, sd3_model)
    
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
        bin_state = torch.load(bin_path, map_location=device)
        
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
    model.to(device)

    print("Model loaded successfully.")
    return model

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    # model checkpoints
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to the seedraw checkpoint to load.")
    parser.add_argument("--output_dir", type=str, default="sd3-dreambooth", help="Directory for model predictions and checkpoints.")

    parser.add_argument("--pretrained_diffusion_ckpt", type=str, required=True, help="Path to the pretrained diffusion checkpoint.")
    parser.add_argument("--pretrained_lmm_ckpt", type=str, required=True, help="Path to the pretrained LMM checkpoint.")

    parser.add_argument("--keep_context_embedder", action="store_true", help="Keep the original context embedder of MMDiT")

    parser.add_argument("--cfg_ratio", type=float, default=None)
    parser.add_argument("--random_vit_skip", action="store_true")




    # dataset 
    parser.add_argument("--dataset_config", type=str, default=None, help="The path to the dataset configuration file.")
    parser.add_argument("--cache_dir", type=str, default="./QwenVL-GEN/diffusers/examples/dreambooth/cache", help="The directory where the downloaded models and datasets will be stored.")

    # training
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible training.")
    parser.add_argument("--output_resolution", type=int, default=512, help="The resolution for input images.")
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop the input images.")
    parser.add_argument("--random_flip", action="store_true", help="Whether to randomly flip images horizontally.")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size per device.")
    parser.add_argument("--sample_batch_size", type=int, default=4, help="Sampling batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates.")

    parser.add_argument("--lora_rank", type=int, default=16, help="Rank for Lora.")
    parser.add_argument("--use_lmm_attention_lora", action="store_true", help="Whether to use lora for LMM attention.")
    parser.add_argument("--use_dit_attention_lora", action="store_true", help="Whether to use lora for DiT attention.")
    parser.add_argument("--use_vit_attention_lora", action="store_true", help="Whether to use lora for ViT attention.")

    parser.add_argument("--use_lmm_mlp_lora", action="store_true", help="Use LMM lora on MLP.")
    parser.add_argument("--use_dit_mlp_lora", action="store_true", help="Use DIT lora on MLP.")
    parser.add_argument("--dit_lora_layers", type=str, default=None, help="Specify layers for DIT lora.")
    
    parser.add_argument("--unfreeze_lmm", action="store_true", help="Whether to unfreeze the language model head.")
    parser.add_argument("--unfreeze_lmm_layers", type=str, default=None, help="The specific layers to unfreeze in the lmm.")
    parser.add_argument("--unfreeze_dit", action="store_true", help="Whether to unfreeze the DIT.")
    parser.add_argument("--unfreeze_dit_layers", type=str, default=None, help="The specific layers to unfreeze in the DIT.")
    parser.add_argument("--unfreeze_adapter", action="store_true", help="Whether to unfreeze the adapter.")
    parser.add_argument("--unfreeze_dit_condition_branch", action="store_true", help="Whether to unfreeze the DIT condition branch.")
    parser.add_argument("--unfreeze_dit_embed", action="store_true", help="Whether to unfreeze the DIT time embed.")
    parser.add_argument("--unfreeze_dit_output_layer", action="store_true", help="Whether to unfreeze DIT output layer.")
    parser.add_argument("--linear_alignment", action="store_true", help="Whether to use linear alignment.")
    parser.add_argument("--lmm_output_layer_index", type=int, default=-1, help="Layer index for LMM output.")
    parser.add_argument("--input_resolution", type=int, default=512, help="Resolution for input images in training/validation dataset.")
    parser.add_argument("--do_lmm_post_norm", action="store_true", help="Whether to use LMM post normalization.")
    parser.add_argument("--max_sequence_length", type=int, default=None, help="Maximum sequence length for T5 text encoder.")

    parser.add_argument("--structure", type=str, default=None)

    # evaluation
    parser.add_argument("--validation_prompts", type=str, default=None, help="Prompts used in validation for learning verification.")
    parser.add_argument("--validation_images", type=str, default="./diffusers/examples/dreambooth/syncdog_val.png", help="Image used during validation.")
    parser.add_argument("--num_validation_images", type=int, default=1, help="Number of images generated during validation with `validation_prompt`.")
    parser.add_argument("--validation_steps", type=int, default=500, help="Run DreamBooth validation every X steps.")

    parser.add_argument("--validation_edit_prompts", type=str, default=None)
    parser.add_argument("--validation_edit_images", type=str, default=None)

    

    # others
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Steps to accumulate before backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")

    parser.add_argument("--lr_scheduler", type=str, default="constant", help="Scheduler type for learning rate.")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of cycles for cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=16, help="Number of subprocesses for data loading.")
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"], help="Scheme for weighting.")
    parser.add_argument("--logit_mean", type=float, default=0.0, help="Mean for logit_normal weighting scheme.")
    parser.add_argument("--logit_std", type=float, default=1.0, help="Standard deviation for logit_normal weighting scheme.")
    parser.add_argument("--mode_scale", type=float, default=1.29, help="Scale for mode weighting scheme.")

    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer type.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 for Adam/Prodigy optimizers.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 for Adam/Prodigy optimizers.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4, help="Weight decay for UNet parameters.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam/Prodigy optimizers.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard log directory.")
    parser.add_argument("--report_to", type=str, default="wandb", help="Integration to report logs to; can be 'tensorboard', 'wandb', or 'comet_ml'.")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="Mode for mixed precision training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank.")

    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args



def main(args):

    # prepare accelerate distributed training

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    
        
    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_diffusion_ckpt, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_diffusion_ckpt,
        subfolder="vae",
    )

    processor = AutoProcessor.from_pretrained(args.pretrained_lmm_ckpt, max_pixels = args.input_resolution*28*28)
    qwenvl2_model = Qwen2VLForConditionalGeneration.from_pretrained(args.pretrained_lmm_ckpt)
    sd3_model = SD3Transformer2DModel.from_pretrained(args.pretrained_diffusion_ckpt, subfolder="transformer")

    if args.structure=="direct":
        print("use direct structure")
        transformer = QwenVLSD3_DirectMap_Transformer2DModel(qwenvl2_model, sd3_model, linear_alignment=args.linear_alignment, lmm_output_layer_index=args.lmm_output_layer_index, do_lmm_post_norm=args.do_lmm_post_norm)
    else:
        transformer = QwenVLSD3Transformer2DModel(qwenvl2_model, sd3_model, linear_alignment=args.linear_alignment, lmm_output_layer_index=args.lmm_output_layer_index, do_lmm_post_norm=args.do_lmm_post_norm)

    transformer.requires_grad_(False)

    # setup lora
    dit_lora_target_modules = []
    lmm_lora_target_modules = []


    if args.use_vit_attention_lora:
        lmm_lora_target_modules += [
            "attn.qkv",
        ]

   
    



    if args.use_dit_attention_lora:
        dit_lora_target_modules += [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]

    if args.use_dit_mlp_lora:
        dit_lora_target_modules += [
                "ff.net.0.proj",
                "ff.net.2.proj",
                "ff_context.net.0.proj",
                "ff_context.net.2.proj"
    ]





    if len(dit_lora_target_modules) > 0:
        logger.info(f"Adding DiT Lora with rank {args.lora_rank} to the following modules: {dit_lora_target_modules}")
        dit_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2*args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=dit_lora_target_modules,
        )
        transformer.dit.add_adapter(dit_lora_config)
   
    if args.use_lmm_attention_lora:
        lmm_lora_target_modules += [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]

    if args.use_lmm_mlp_lora:
        lmm_lora_target_modules += [
                "mlp.down_proj",
                "mlp.gate_proj",
                "mlp.up_proj"
                ]

    if len(lmm_lora_target_modules) > 0:
        logger.info(f"Adding LMM Lora with rank {args.lora_rank} to the following modules: {lmm_lora_target_modules}")
        lmm_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2*args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=lmm_lora_target_modules,
        )

        transformer.lmm.add_adapter(lmm_lora_config)
    
    # resume from checkpoint

    if args.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        transformer = load_sharded_model(
            os.path.join(args.resume_from_checkpoint,"config.json"),
            os.path.join(args.resume_from_checkpoint,"diffusion_pytorch_model.bin.index.json"),
            os.path.join(args.resume_from_checkpoint,),
            qwenvl2_model,
            sd3_model,
            "cpu",
        )
    
    params_to_optimize = []

    # unfreeze parameters

    if args.unfreeze_dit:
        transformer.dit.requires_grad_(True)
    if args.unfreeze_dit_layers:
        layers_to_unfreeze = [int(i) for i in args.unfreeze_dit_layers.split(",")]
        for layer_index in layers_to_unfreeze:
            transformer.dit.transformer_blocks[layer_index].requires_grad_(True)

    
    if args.unfreeze_lmm:
        transformer.lmm.requires_grad_(True)
    if args.unfreeze_lmm_layers:
        layers_to_unfreeze = [int(i) for i in args.unfreeze_lmm_layers.split(",")]
        for layer_index in layers_to_unfreeze:
            transformer.lmm.model.layers[layer_index].requires_grad_(True)

    if args.unfreeze_dit_output_layer:
        transformer.dit.norm_out.requires_grad_(True)
        transformer.dit.proj_out.requires_grad_(True)
    
    if args.unfreeze_dit_embed:
        transformer.dit.time_text_embed.requires_grad_(True)
        transformer.dit.context_embedder.requires_grad_(True)
        transformer.dit.pos_embed.requires_grad_(True)
        

    if args.unfreeze_adapter:
        transformer.input_embeds_align_mlp.requires_grad_(True)
        transformer.condition_embeds_align_mlp.requires_grad_(True)
        transformer.norm_lmm_out.requires_grad_(True)
        transformer.pooled_proj.requires_grad_(True)


    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    params_to_optimize.append({"params": transformer_trainable_parameters, "lr": args.learning_rate})

    
    # Calculate the number of parameters to train and all parameters

    num_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in transformer.parameters())

    
    logger.info(f"Number of parameters: {num_params}")
    logger.info(f"Number of parameters to train: {num_trainable_params}")

    # ratio
    logger.info(f"Trainable parameters ratio: {num_trainable_params / num_params:.5f}")

    # setting weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=torch.float32)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"),safe_serialization=False)
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()
            # load diffusers style into model
            if isinstance(unwrap_model(model), SD3Transformer2DModel):
                load_model = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())

            else:
                raise ValueError(f"Unsupported model found: {type(model)=}")
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


       
    # set up optimizer
    optimizer = torch.optim.AdamW(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    
    # set up dataset

    with open(args.dataset_config, "r") as f:
        dataset_config = json.load(f)
    
    datasets = []
    datasets_lengths = []

    for subdataset_config in dataset_config:
        datasets.append(InterleavedDataset(subdataset_config, size=args.output_resolution, center_crop=args.center_crop, cache_dir=args.cache_dir))
        datasets_lengths.append(subdataset_config['number'])
        print(subdataset_config)
    
    train_dataset = ConcatDataset(datasets, datasets_lengths)
    

    # data loader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    print("num_update_steps_per_epoch",num_update_steps_per_epoch)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )


    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )
    
     # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print("num_update_steps_per_epoch",num_update_steps_per_epoch)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "dreambooth-sd3"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    transformer.train()

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                pixel_values = batch["tensor_image"].to(dtype=vae.dtype)
                prompts = batch["prompt"]
                images = batch["pil_image"]
                tasks = batch["task"]

                

                

                # process input for different tasks

                assert len(tasks) == 1, "Only one task is supported for now."


                if tasks[0] == "T2I":
                    
                    text = ["Generate Image: " + prompt for prompt in prompts]

                    if args.cfg_ratio and random.random() <= args.cfg_ratio:
                        # drop the prompt according to the ratio
                        text = ["Generate Image: " for prompt in text]
                    
                    
                    
                    
                    if args.max_sequence_length is not None:
                        inputs = processor(
                            text=text,
                            padding="max_length",
                            return_tensors="pt",
                            max_length=args.max_sequence_length,
                            truncation=True,
                        )
                    else:
                        inputs = processor(
                            text=text,
                            padding="longest",
                            return_tensors="pt",
                    )
                
                elif tasks[0] == "I2I":
                    
                    messages = [
                        {
                        "role": "user",
                        "content":[
                            {
                                "type": "image",
                                "image": img
                            }]
                        } for img in images
                    ]

                    # text = processor.apply_chat_template(
                    #     messages, tokenize=False
                    # )

                    text = ["Copy Image: <|vision_start|><|image_pad|><|vision_end|>" for img in images]

                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = processor(
                    text=text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    )



                elif tasks[0] == "TI2I":

                    pixel_values = batch["target_image"].to(dtype=vae.dtype)

                    messages = [
                        {
                        "role": "user",
                        "content":[
                            {
                                "type": "image",
                                "image": img
                            }]
                        } for img in images
                    ]

                    text = ["Edit image <|vision_start|><|image_pad|><|vision_end|> following:" + prompt for prompt in  prompts]

                    if args.cfg_ratio and random.random() <= args.cfg_ratio:
                        # drop the prompt according to the ratio
                        text = ["Edit image <|vision_start|><|image_pad|><|vision_end|> following: " for prompt in text]

                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = processor(
                    text=text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    )
                
                elif tasks[0] == "OBJ2I":

                    pixel_values = batch["target_image"].to(dtype=vae.dtype)
                    obj_names = batch['obj_name_image']

                    # import pdb; pdb.set_trace()

                    messages = [
                        {
                        "role": "user",
                        "content":[
                            {
                                "type": "image",
                                "image": obj[1]
                            } for obj in obj_names[0]
                        ]   
                        }
                    ]

                    image_caption = prompts[0]
                    obj_names_text =  ", ".join([obj[0]+"<|vision_start|><|image_pad|><|vision_end|>" for obj in obj_names[0]])



                    text = ["Combine the objects: "+obj_names_text+" and generate an image following: "+image_caption]

                    if args.cfg_ratio and random.random() <= args.cfg_ratio:
                        # drop the prompt according to the ratio
                        text = ["Combine the objects: "+obj_names_text+" and generate an image following: "]

                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = processor(
                    text=text,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    )

                elif tasks[0] == "OBJ2I_V2":

                    pixel_values = batch["target_image"].to(dtype=vae.dtype)
                    obj_names = batch['obj_name_image']

                    

                    messages = [
                        {
                        "role": "user",
                        "content":[
                            {
                                "type": "image",
                                "image": obj
                            } for obj in obj_names[0] if isinstance(obj, Image.Image)
                        ]   
                        }
                    ]

                    if args.cfg_ratio and random.random() <= args.cfg_ratio:
                    # drop the images according to the ratio
                        image_caption = [i  for i in obj_names[0] if isinstance(i, str) ]
                        text = ["Generate Image: "+"".join(image_caption).strip()]

                        inputs = processor(
                        text=text,
                        padding=True,
                        return_tensors="pt",
                        )
                    else:
                        image_caption = [i if isinstance(i, str) else "<|vision_start|><|image_pad|><|vision_end|>" for i in obj_names[0]]
                        text = ["Generate Image: "+"".join(image_caption).strip()]
                        image_inputs, video_inputs = process_vision_info(messages)

                        inputs = processor(
                        text=text,
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                        )

                    



                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # HACK unify inputs devices
                inputs = inputs.to(device=model_input.device, dtype=weight_dtype)
                inputs = {f"lmm_{k}": v for k, v in inputs.items()} 

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise


                inputs["dit_hidden_states"] = noisy_model_input
                inputs["dit_time_step"] = timesteps

                if args.random_vit_skip:
                    inputs["vit_skip_ratio"] = torch.rand(bsz, device=model_input.device, dtype=weight_dtype)


           
                if args.keep_context_embedder:
                    model_pred = transformer.forward_with_original_context_embedder(**inputs)[0]
                else:
                    model_pred = transformer(
                        **inputs
                    )[0]

                
                

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if args.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

              

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()


                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                

                    

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:

                    if global_step % args.validation_steps == 0:

                        if args.validation_prompts is not None:
                    # create pipeline

                            pipeline = QwenVLStableDiffusion3Pipeline(
                                accelerator.unwrap_model(transformer),
                                processor,
                                noise_scheduler,
                                vae,
                            )

                            test_prompts = args.validation_prompts.split("|")
                            
                            if args.max_sequence_length is not None:
                                pipeline_args_list = [{"prompt": prompt, "width":args.output_resolution, "height":args.output_resolution, "max_sequence_length":args.max_sequence_length} for prompt in test_prompts]
                            else:
                                pipeline_args_list = [{"prompt": prompt, "width":args.output_resolution, "height":args.output_resolution} for prompt in test_prompts]
                            
                            images = log_validation(
                                pipeline=pipeline,
                                args=args,
                                accelerator=accelerator,
                                pipeline_args_list=pipeline_args_list,
                                epoch=global_step,
                                torch_dtype=weight_dtype,
                            )
                            
                        if args.validation_images is not None:

                            pipeline = QwenVLStableDiffusion3Pipeline(
                                accelerator.unwrap_model(transformer),
                                processor,
                                noise_scheduler,
                                vae,
                            )

                            test_images = []

                            for test_img_path in args.validation_images.split(","):
                                test_img = Image.open(test_img_path)
                                test_transform = transforms.Compose(
                                    [
                                        transforms.Resize(args.output_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                                        transforms.CenterCrop(args.output_resolution),
                                    ]
                                )
                                test_img = test_transform(test_img)
                                test_images.append(test_img)
                            
                        
                            pipeline_args_list = [{"image": img, "width":args.output_resolution, "height":args.output_resolution} for img in test_images]
                            
                            
                            images = log_validation(
                                pipeline=pipeline,
                                args=args,
                                accelerator=accelerator,
                                pipeline_args_list=pipeline_args_list,
                                epoch=global_step,
                                torch_dtype=weight_dtype,
                            )

                        if args.validation_edit_prompts is not None and args.validation_edit_images is not None:
                            pipeline = QwenVLStableDiffusion3Pipeline(
                                accelerator.unwrap_model(transformer),
                                processor,
                                noise_scheduler,
                                vae,
                            )

                            test_prompts = args.validation_edit_prompts.split("|")
                            test_images = []

                            for test_img_path in args.validation_edit_images.split(","):
                                test_img = Image.open(test_img_path)
                                test_transform = transforms.Compose(
                                    [
                                        transforms.Resize(args.output_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                                        transforms.CenterCrop(args.output_resolution),
                                    ]
                                )
                                test_img = test_transform(test_img)
                                test_images.append(test_img)
                            
                            pipeline_args_list = [{"prompt": prompt, "image": img, "width":args.output_resolution, "height":args.output_resolution, "max_sequence_length":400} for prompt, img in zip(test_prompts, test_images)]
                            
                            images = log_validation(
                                pipeline=pipeline,
                                args=args,
                                accelerator=accelerator,
                                pipeline_args_list=pipeline_args_list,
                                epoch=global_step,
                                torch_dtype=weight_dtype
                            )
                            

                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            
                
                



      

        

    accelerator.end_training()




if __name__ == "__main__":
    args = parse_args()
    main(args)
