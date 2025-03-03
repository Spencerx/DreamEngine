# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union


from diffusers.models.transformers.transformer_sd3 import QwenVLSD3Transformer2DModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image



import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from ...image_processor import VaeImageProcessor
from ...loaders import FromSingleFileMixin, SD3LoraLoaderMixin
from ...models.autoencoders import AutoencoderKL
from ...models.transformers import SD3Transformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import StableDiffusion3PipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
        ```
"""


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class QwenVLStableDiffusion3Pipeline(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin):
    r"""
    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
    """
    def __init__(
        self,
        transformer: QwenVLSD3Transformer2DModel,
        processor: AutoProcessor,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
    ):
        super().__init__()

        self.register_modules(
            processor=processor,
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
        self.default_sample_size = (
            self.transformer.dit.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.patch_size = (
            self.transformer.dit.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[str] = None,
        prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,       
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_sequence_length: Optional[int] = None,
        vit_skip_ratio: Optional[float] = None,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._interrupt = False

        # 2. Define call parameters
        
        batch_size = 1

        device = self.transformer.dit.device

        # 3. Prepare prompt embeddings

        if image and prompt==None:

            messages = [
                    {
                    "role": "user",
                    "content":[
                        {
                            "type": "image",
                            "image": image
                        }]
                    } 
                ]

            text = "Copy Image: <|vision_start|><|image_pad|><|vision_end|>" 

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            )

        elif prompt and image==None:

            messages = [
                    {
                    "role": "user",
                    "content":[
                        {
                            "type": "text",
                            "text": prompt
                        }]
                    } 
                ]

            text = self.processor.apply_chat_template(
                    messages, tokenize=False
                )

            if max_sequence_length is not None:
                inputs = self.processor(
                    text=[text],
                    padding="max_length",
                    return_tensors="pt",
                    max_length=max_sequence_length,
                    truncation=True
                )

            else:
                inputs = self.processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                )

        else:
            raise ValueError("You must provide either an image or a prompt.")
            

        

        inputs = inputs.to(device=self.transformer.dit.device, dtype=self.transformer.dit.dtype)
        inputs = {f"lmm_{k}": v for k, v in inputs.items()} 

        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.dit.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.transformer.dit.dtype,
            device,
            generator,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                inputs["dit_hidden_states"] = latent_model_input
                inputs["dit_time_step"] = timestep

                inputs["vit_skip_ratio"] = vit_skip_ratio

                noise_pred = self.transformer(
                    **inputs,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)

    @torch.no_grad()
    def cfg_predict(
        self,
        image: Optional[str] = None,
        prompt: Optional[str] = None,
        segments: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,       
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        max_sequence_length: Optional[int] = 77,
        guidance_scale: Optional[float] = 1.0,
        vit_skip_ratio: Optional[float] = None,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._interrupt = False

        # 2. Define call parameters
        
        batch_size = 1

        device = self.transformer.dit.device

        # 3. Prepare prompt embeddings

        if image and prompt==None:

            messages = [
                    {
                    "role": "user",
                    "content":[
                        {
                            "type": "image",
                            "image": image
                        }]
                    } 
                ]

            text = "Copy Image: <|vision_start|><|image_pad|><|vision_end|>" 

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            return_tensors="pt",
            max_length=max_sequence_length,
            )

            

            null_text = "Copy Image: "
            
            null_inputs = self.processor(
            text=null_text,
            padding="max_length",
            return_tensors="pt",
            max_length=max_sequence_length,
            truncation=True
            )



        elif prompt and image==None and segments==None:

            text = "Generate Image: "+prompt

            if max_sequence_length is not None:
                inputs = self.processor(
                    text=[text],
                    padding="max_length",
                    return_tensors="pt",
                    max_length=max_sequence_length,
                    truncation=True
                )

            else:
                inputs = self.processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt",
                )

            
            

            null_text = "Generate Image: "
            
            null_inputs = self.processor(
                    text=null_text,
                    padding="max_length",
                    return_tensors="pt",
                    max_length=max_sequence_length,
                    truncation=True
            )

        elif prompt and image:

            messages = [
                    {
                    "role": "user",
                    "content":[
                        {
                            "type": "image",
                            "image": image
                        }]
                    } 
                ]

            text = "Edit image <|vision_start|><|image_pad|><|vision_end|> following:" + prompt  

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            return_tensors="pt",
            max_length=max_sequence_length
            )

            null_text = "Edit image <|vision_start|><|image_pad|><|vision_end|> following:"
            
            null_inputs = self.processor(
            text=[null_text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            return_tensors="pt",
            max_length=max_sequence_length
            )

        elif prompt and segments:
            
            obj_names = segments

                # import pdb; pdb.set_trace()

            messages = [
                {
                "role": "user",
                "content":[
                    {
                        "type": "image",
                        "image": obj[1]
                    } for obj in obj_names
                ]   
                }
            ]

            image_caption = prompt
            obj_names_text =  ", ".join([obj[0]+"<|vision_start|><|image_pad|><|vision_end|>" for obj in obj_names])



            text = ["Combine the objects: "+obj_names_text+" and generate an image following: "+image_caption]

            image_inputs, video_inputs = process_vision_info(messages)

            text_token_length = len(self.processor.tokenizer(text)['input_ids'][0])
            max_sequence_length = int(text_token_length+image_inputs[0].size[0]*image_inputs[0].size[1]/28/28+image_inputs[1].size[0]*image_inputs[1].size[1]/28/28-2)


            inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            max_length=max_sequence_length
            )

            null_text = ["Combine the objects: "+obj_names_text+" and generate an image following: "]

            null_inputs = self.processor(
            text=null_text,
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            return_tensors="pt",
            max_length=max_sequence_length
            )

        elif prompt==None and segments:
            
            obj_names = segments

                    
            messages = [
                {
                "role": "user",
                "content":[
                    {
                        "type": "image",
                        "image": obj
                    } for obj in obj_names if isinstance(obj, Image.Image)
                ]   
                }
            ]

            image_caption = [i if isinstance(i, str) else "<|vision_start|><|image_pad|><|vision_end|>" for i in obj_names]
            text = ["Generate Image: "+"".join(image_caption).strip()]
            image_inputs, video_inputs = process_vision_info(messages)





            inputs = self.processor(
                                    text=text,
                                    images=image_inputs,
                                    videos=video_inputs,
                                    padding=True,
                                    return_tensors="pt",
                                    max_length=max_sequence_length
                                    )

            image_caption = [i  for i in obj_names if isinstance(i, str) ]
            null_text = ["Generate Image: "+"".join(image_caption).strip()]

            null_inputs = self.processor(
            text=null_text,
            padding="max_length",
            return_tensors="pt",
            max_length=max_sequence_length
            )
                    


            



        else:
            raise ValueError("You must provide either an image or a prompt.")
            

        

        inputs = inputs.to(device=self.transformer.dit.device, dtype=self.transformer.dit.dtype)
        inputs = {f"lmm_{k}": v for k, v in inputs.items()} 

        null_inputs = null_inputs.to(device=self.transformer.dit.device, dtype=self.transformer.dit.dtype)
        null_inputs = {f"lmm_{k}": v for k, v in null_inputs.items()}

        # cat all the input and null inputs for cfg prediction

        inputs["lmm_input_ids"] = torch.cat([inputs["lmm_input_ids"], null_inputs["lmm_input_ids"]])
        inputs["lmm_attention_mask"] = torch.cat([inputs["lmm_attention_mask"], null_inputs["lmm_attention_mask"]])

        if prompt and image:
            inputs["lmm_pixel_values"] = torch.cat([inputs["lmm_pixel_values"], null_inputs["lmm_pixel_values"]])
            inputs["lmm_image_grid_thw"] = torch.cat([inputs["lmm_image_grid_thw"], null_inputs["lmm_image_grid_thw"]])
        
        if prompt and segments:
            inputs["lmm_pixel_values"] = torch.cat([inputs["lmm_pixel_values"], null_inputs["lmm_pixel_values"]])
            inputs["lmm_image_grid_thw"] = torch.cat([inputs["lmm_image_grid_thw"], null_inputs["lmm_image_grid_thw"]])



 


        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.dit.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.transformer.dit.dtype,
            device,
            generator,
        )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance

                latent_model_input = torch.cat([latents] * 2) 


                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                inputs["dit_hidden_states"] = latent_model_input
                inputs["dit_time_step"] = timestep

                inputs["vit_skip_ratio"] = vit_skip_ratio


                noise_pred = self.transformer(
                    **inputs,
                )[0]

                noise_pred_text, noise_pred_uncond  = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)



                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)
