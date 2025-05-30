# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
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


from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.attention import JointTransformerBlock
from ...models.attention_processor import Attention, AttentionProcessor, FusedJointAttnProcessor2_0
from ...models.modeling_utils import ModelMixin
from ...models.normalization import AdaLayerNormContinuous
from ...utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from ..embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from ..modeling_outputs import Transformer2DModelOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"




class QwenVLSD3_DirectMap_Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    def __init__(self, qwenvl_model, sd3_dit_model, mlp_dim=4096, linear_alignment=False, lmm_output_layer_index=-1, do_lmm_post_norm=False):
        super().__init__()

        self.lmm = qwenvl_model
        self.dit = sd3_dit_model

        self.lmm_output_layer_index = lmm_output_layer_index

        self.do_lmm_post_norm = do_lmm_post_norm
        self.norm_lmm_out = LlamaRMSNorm(self.lmm.config.hidden_size)

        self.pooled_proj = nn.Linear(4096,2048)

        if not linear_alignment:
            self.input_embeds_align_mlp = nn.Sequential(
                nn.Linear(self.lmm.config.hidden_size, mlp_dim),
                nn.SiLU(),
                nn.Linear(mlp_dim, self.dit.config.caption_projection_dim)
            )

            self.condition_embeds_align_mlp = nn.Sequential(
                nn.Linear(self.lmm.config.hidden_size, mlp_dim),
                nn.SiLU(),
                nn.Linear(mlp_dim, self.dit.config.caption_projection_dim)
            )
        else:
            self.input_embeds_align_mlp = nn.Linear(self.lmm.config.hidden_size, self.dit.config.caption_projection_dim)
            self.condition_embeds_align_mlp = nn.Linear(self.lmm.config.hidden_size, self.dit.config.caption_projection_dim)
            
            
            
    
    def forward(self, lmm_input_ids, lmm_attention_mask, dit_hidden_states, dit_time_step, dit_text_condition=None,pooled_dit_text_condition=None,  lmm_pixel_values=None, lmm_image_grid_thw=None, vit_skip_ratio=None):
        lmm_outputs_last_hidden_state = self.lmm(
            input_ids=lmm_input_ids,
            attention_mask=lmm_attention_mask,
            pixel_values=lmm_pixel_values,
            image_grid_thw=lmm_image_grid_thw,
            output_hidden_states=True
        )['hidden_states'][self.lmm_output_layer_index]

        if vit_skip_ratio and isinstance(lmm_pixel_values,torch.Tensor):
            vit_feature = self.lmm.visual(lmm_pixel_values,lmm_image_grid_thw)
            image_mask = lmm_input_ids == self.lmm.config.image_token_id
            # extract the masked hidden stages and add the vit features according to the ratio (1-vit_skip_ratio)*hidden_states + vit_skip_ratio*vit_features
            image_hidden_states = lmm_outputs_last_hidden_state[image_mask]
            blended_features = (1 - vit_skip_ratio) * image_hidden_states + vit_skip_ratio * vit_feature
            lmm_outputs_last_hidden_state[image_mask] = blended_features
        


        if self.do_lmm_post_norm:
            lmm_outputs_last_hidden_state = self.norm_lmm_out(lmm_outputs_last_hidden_state)

        

        # Encoder Hidden States and Pooled Hidden States
        dit_encoder_hidden_states_proj = self.input_embeds_align_mlp(lmm_outputs_last_hidden_state)
        dit_encoder_hidden_states_pooled_proj = self.condition_embeds_align_mlp(lmm_outputs_last_hidden_state.mean(dim=1))

        # Time Condition
        dit_time_embed = self.dit.time_text_embed.time_proj(dit_time_step)
        dit_timesteps_emb = self.dit.time_text_embed.timestep_embedder(dit_time_embed.to(dtype=dit_encoder_hidden_states_pooled_proj.dtype))
        dit_concat_condition = dit_encoder_hidden_states_pooled_proj + dit_timesteps_emb

        dit_model_pred = self.dit.forward_with_lmm_encoder(
                        hidden_states=dit_hidden_states,
                        lmm_encoder_hidden_states=dit_encoder_hidden_states_proj,
                        lmm_pooled_projections=dit_concat_condition,
                        dit_text_condition=dit_text_condition,
                        return_dict=False,
                    )
        
        return dit_model_pred
    


class QwenVLSD3Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    def __init__(self, qwenvl_model, sd3_dit_model, mlp_dim=4096, linear_alignment=False, lmm_output_layer_index=-1, do_lmm_post_norm=False):
        super().__init__()

        # print("QwenVLSD3Transformer2DModel: qwenvl_model", qwenvl_model)
        # print("QwenVLSD3Transformer2DModel: sd3_dit_model", sd3_dit_model)
        # print("QwenVLSD3Transformer2DModel: mlp_dim", mlp_dim)
        # print("QwenVLSD3Transformer2DModel: linear_alignment", linear_alignment)
        # print("QwenVLSD3Transformer2DModel: lmm_output_layer_index", lmm_output_layer_index)
        # print("QwenVLSD3Transformer2DModel: do_lmm_post_norm", do_lmm_post_norm)

        self.lmm = qwenvl_model
        self.dit = sd3_dit_model

        self.lmm_output_layer_index = lmm_output_layer_index

        self.do_lmm_post_norm = do_lmm_post_norm
        self.norm_lmm_out = LlamaRMSNorm(self.lmm.config.hidden_size)


        self.pooled_proj = nn.Linear(4096,2048)

        if not linear_alignment:
            self.input_embeds_align_mlp = nn.Sequential(
                nn.Linear(self.lmm.config.hidden_size, 8192),
                nn.SiLU(),
                nn.Linear(8192, 4096)
            )

            self.condition_embeds_align_mlp = nn.Sequential(
                nn.Linear(self.lmm.config.hidden_size,8192),
                nn.SiLU(),
                nn.Linear(8192,4096)
            )
        else:
            self.input_embeds_align_mlp = nn.Linear(self.lmm.config.hidden_size, 4096)
            self.condition_embeds_align_mlp = nn.Linear(self.lmm.config.hidden_size, 4096)

            
    def forward(self, lmm_input_ids, lmm_attention_mask, dit_hidden_states, dit_time_step, dit_text_condition=None,pooled_dit_text_condition=None,  lmm_pixel_values=None, lmm_image_grid_thw=None, vit_skip_ratio=None):
        lmm_outputs_last_hidden_state = self.lmm(
            input_ids=lmm_input_ids,
            attention_mask=lmm_attention_mask,
            pixel_values=lmm_pixel_values,
            image_grid_thw=lmm_image_grid_thw,
            output_hidden_states=True
        )['hidden_states'][self.lmm_output_layer_index]


        if self.do_lmm_post_norm:
            lmm_outputs_last_hidden_state = self.norm_lmm_out(lmm_outputs_last_hidden_state)

        # align layer
        dit_encoder_hidden_states_proj = self.input_embeds_align_mlp(lmm_outputs_last_hidden_state)

        

        

        
        dit_model_pred = self.dit(
                        hidden_states=dit_hidden_states,
                        encoder_hidden_states=dit_encoder_hidden_states_proj,
                        pooled_projections=self.pooled_proj(dit_encoder_hidden_states_proj.mean(dim=1)),
                        timestep = dit_time_step,
                        return_dict=False,
                    )
    
        return dit_model_pred

    
    # def forward(self, lmm_input_ids, lmm_attention_mask, dit_hidden_states, dit_time_step, dit_text_condition=None,pooled_dit_text_condition=None,  lmm_pixel_values=None, lmm_image_grid_thw=None, vit_skip_ratio=None):
    #     lmm_outputs_last_hidden_state = self.lmm(
    #         input_ids=lmm_input_ids,
    #         attention_mask=lmm_attention_mask,
    #         pixel_values=lmm_pixel_values,
    #         image_grid_thw=lmm_image_grid_thw,
    #         output_hidden_states=True
    #     )['hidden_states'][self.lmm_output_layer_index]


    #     if self.do_lmm_post_norm:
    #         lmm_outputs_last_hidden_state = self.norm_lmm_out(lmm_outputs_last_hidden_state)



    #     # Encoder Hidden States and Pooled Hidden States
    #     dit_encoder_hidden_states_proj = self.input_embeds_align_mlp(lmm_outputs_last_hidden_state)
    #     dit_encoder_hidden_states_pooled_proj = self.condition_embeds_align_mlp(lmm_outputs_last_hidden_state.mean(dim=1))

    #     # Time Condition
    #     if pooled_dit_text_condition is not None:
    #         temb = self.dit.time_text_embed(dit_time_step, pooled_dit_text_condition) + dit_encoder_hidden_states_pooled_proj
    #         dit_concat_condition = dit_encoder_hidden_states_pooled_proj + temb
    #     else:
    #         dit_time_embed = self.dit.time_text_embed.time_proj(dit_time_step)
    #         dit_timesteps_emb = self.dit.time_text_embed.timestep_embedder(dit_time_embed.to(dtype=dit_encoder_hidden_states_pooled_proj.dtype))
    #         dit_concat_condition = dit_encoder_hidden_states_pooled_proj + dit_timesteps_emb

    #     dit_model_pred = self.dit.forward_with_lmm_encoder(
    #                     hidden_states=dit_hidden_states,
    #                     lmm_encoder_hidden_states=dit_encoder_hidden_states_proj,
    #                     lmm_pooled_projections=dit_concat_condition,
    #                     dit_text_condition=dit_text_condition,
    #                     return_dict=False,
    #                 )
        
    #     return dit_model_pred

class LlavaSD3Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    def __init__(self, llava_model, sd3_dit_model, mlp_dim=4096, linear_alignment=False, lmm_output_layer_index=-1, do_lmm_post_norm=False):
        super().__init__()

        print("LlavaSD3Transformer2DModel: qwenvl_model", llava_model)
        print("LlavaSD3Transformer2DModel: sd3_dit_model", sd3_dit_model)
        print("LlavaSD3Transformer2DModel: mlp_dim", mlp_dim)
        print("LlavaSD3Transformer2DModel: linear_alignment", linear_alignment)
        print("LlavaSD3Transformer2DModel: lmm_output_layer_index", lmm_output_layer_index)
        print("LlavaSD3Transformer2DModel: do_lmm_post_norm", do_lmm_post_norm)

        self.lmm = llava_model
        self.dit = sd3_dit_model

        self.lmm_output_layer_index = lmm_output_layer_index

        self.do_lmm_post_norm = do_lmm_post_norm
        self.norm_lmm_out = LlamaRMSNorm(4096)

        if not linear_alignment:
            self.input_embeds_align_mlp = nn.Sequential(
                nn.Linear(4096, mlp_dim),
                nn.SiLU(),
                nn.Linear(mlp_dim, self.dit.config.caption_projection_dim)
            )

            self.condition_embeds_align_mlp = nn.Sequential(
                nn.Linear(4096, mlp_dim),
                nn.SiLU(),
                nn.Linear(mlp_dim, self.dit.config.caption_projection_dim)
            )
        else:
            self.input_embeds_align_mlp = nn.Linear(4096, self.dit.config.caption_projection_dim)
            self.condition_embeds_align_mlp = nn.Linear(4096, self.dit.config.caption_projection_dim)
            


    
    def forward(self, lmm_input_ids, lmm_attention_mask, lmm_pixel_values, dit_hidden_states, dit_time_step, dit_text_condition=None,pooled_dit_text_condition=None):
        lmm_outputs_last_hidden_state = self.lmm(
            input_ids=lmm_input_ids,
            attention_mask=lmm_attention_mask,
            pixel_values=lmm_pixel_values,
            output_hidden_states=True
        )['hidden_states'][self.lmm_output_layer_index]

       
        if self.do_lmm_post_norm:
            lmm_outputs_last_hidden_state = self.norm_lmm_out(lmm_outputs_last_hidden_state)

        

        # Encoder Hidden States and Pooled Hidden States
        dit_encoder_hidden_states_proj = self.input_embeds_align_mlp(lmm_outputs_last_hidden_state)
        dit_encoder_hidden_states_pooled_proj = self.condition_embeds_align_mlp(lmm_outputs_last_hidden_state.mean(dim=1))

        # Time Condition
        if pooled_dit_text_condition is not None:
            temb = self.dit.time_text_embed(dit_time_step, pooled_dit_text_condition) + dit_encoder_hidden_states_pooled_proj
            dit_concat_condition = dit_encoder_hidden_states_pooled_proj + temb
        else:
            dit_time_embed = self.dit.time_text_embed.time_proj(dit_time_step)
            dit_timesteps_emb = self.dit.time_text_embed.timestep_embedder(dit_time_embed.to(dtype=dit_encoder_hidden_states_pooled_proj.dtype))
            dit_concat_condition = dit_encoder_hidden_states_pooled_proj + dit_timesteps_emb

        dit_model_pred = self.dit.forward_with_lmm_encoder(
                        hidden_states=dit_hidden_states,
                        lmm_encoder_hidden_states=dit_encoder_hidden_states_proj,
                        lmm_pooled_projections=dit_concat_condition,
                        dit_text_condition=dit_text_condition,
                        return_dict=False,
                    )
        
        return dit_model_pred

        
    




class SD3Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`): Number of dimensions to use when projecting the `encoder_hidden_states`.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        out_channels (`int`, defaults to 16): Number of output channels.

    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: Tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)

        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                    qk_norm=qk_norm,
                    use_dual_attention=True if i in dual_attention_layers else False,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedJointAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedJointAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward_with_lmm_encoder(
        self,
        hidden_states: torch.FloatTensor,
        lmm_encoder_hidden_states: torch.FloatTensor = None,
        lmm_pooled_projections: torch.FloatTensor = None,
        dit_text_condition: torch.FloatTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:


        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = lmm_pooled_projections
        
        
        encoder_hidden_states = lmm_encoder_hidden_states

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
        

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
