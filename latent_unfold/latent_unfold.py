 '''
 Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 '''

import os
import math
from collections import defaultdict
import numpy as np
import random
from typing import Any, Dict, Optional, Tuple, Union, List, Callable
import time

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from diffusers.pipelines.flux.pipeline_flux import (
    replace_example_docstring,
    EXAMPLE_DOC_STRING,
    XLA_AVAILABLE,
    FluxPipelineOutput
)
from diffusers.pipelines.flux.pipeline_flux_inpaint import (
    FluxInpaintPipeline, 
    retrieve_latents,
    retrieve_timesteps,
    calculate_shift,
)
from diffusers.utils import (
    logging,
    is_torch_version,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.image_processor import PipelineImageInput
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_processor import Attention
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


logger = logging.get_logger(__name__)

attn_maps = {}


class LatentUnfoldPipeline(FluxInpaintPipeline):
    def __init__(self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
        )
    
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 0.6,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        mosaic_shape=None,
        seed=0
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            mask_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to mask `image`. White pixels in the mask
                are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
                single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one
                color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B,
                H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W,
                1)`, or `(H, W)`.
            mask_image_latent (`torch.Tensor`, `List[torch.Tensor]`):
                `Tensor` representing an image batch to mask `image` generated by VAE. If not provided, the mask
                latents tensor will ge generated by `mask_image`.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            padding_mask_crop (`int`, *optional*, defaults to `None`):
                The size of margin in the crop to be applied to the image and masking. If `None`, no crop is applied to
                image and mask_image. If `padding_mask_crop` is not `None`, it will first find a rectangular region
                with the same aspect ration of the image and contains all masked area, and then expand that area based
                on `padding_mask_crop`. The image and mask_image will then be cropped based on the expanded area before
                resizing to the original image size for inpainting. This is useful when the masked area is small while
                the image is large and contain information irrelevant for inpainting, such as background.
            strength (`float`, *optional*, defaults to 1.0):
                Indicates extent to transform the reference `image`. Must be between 0 and 1. `image` is used as a
                starting point and more noise is added the higher the `strength`. The number of denoising steps depends
                on the amount of noise initially added. When `strength` is 1, added noise is maximum and the denoising
                process runs for the full number of iterations specified in `num_inference_steps`. A value of 1
                essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        random.seed(seed)
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            mask_image,
            strength,
            height,
            width,
            output_type=output_type,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            padding_mask_crop=padding_mask_crop,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Preprocess mask and image
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        init_images = []
        for img in image:
            init_image = self.image_processor.preprocess(
                img, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
            )
            init_image = init_image.to(dtype=torch.float32)
            init_images.append(init_image)

        # 3. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4.Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        num_channels_transformer = self.transformer.config.in_channels

        latents, noise, image_latents, latent_image_ids, mask = self.prepare_mosaic_latent(
            init_images,
            latent_timestep,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            mosaic_shape=mosaic_shape,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    ##################################################
                    height=height, step=i,
                    ##################################################
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # for 64 channel transformer only.
                init_latents_proper = image_latents
                init_mask = mask

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.scale_noise(
                        init_latents_proper, torch.tensor([noise_timestep]), noise
                    )
                latents = (1 - init_mask) * init_latents_proper + init_mask * latents
                    

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            mosaic_height, mosaic_width = height * mosaic_shape[0], width * mosaic_shape[1]
            latents = self._unpack_latents(latents, mosaic_height, mosaic_width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents[:,:,:latents.shape[2] // mosaic_shape[0],:latents.shape[3] // mosaic_shape[1]]

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


    def prepare_mosaic_latent(
        self,
        images,
        timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        mosaic_shape=(3,3),      
        ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        # print(latent_height, latent_width) # 64, 64
        # shape = (batch_size, num_channels_latents, latent_height, latent_width)
        
        image_latents = []
        image = None
        for i, image in enumerate(images):
            image = image.to(device=device, dtype=dtype)
            image_latent = self._encode_vae_image(image=image, generator=generator)
            image_latents.append(image_latent)
        # print(image.shape) # torch.Size([1, 3, 512, 512])
        white_image = torch.ones_like(image, device=image.device, dtype=image.dtype)
        white_image_latent = self._encode_vae_image(image=white_image, generator=generator)

        # unfold to make mosaic
        mosaic_latent_height, mosaic_latent_width = latent_height * mosaic_shape[0], latent_width * mosaic_shape[1]
        shape = (batch_size, num_channels_latents, mosaic_latent_height, mosaic_latent_width)
        # print(mosaic_latent_height, mosaic_latent_width) # 192, 192
        # print(shape) # (1, 16, 192, 192)
        mosaic_image_latent = torch.tile(white_image_latent, (1, 1, mosaic_shape[0], mosaic_shape[1]))
        # print(mosaic_image_latent.shape) # torch.Size([1, 16, 192, 192])

        grid_count = mosaic_shape[0] * mosaic_shape[1]
        image_latents = image_latents[:grid_count - 1]
        for m in range(mosaic_shape[0]):
            for n in range(mosaic_shape[1]):
                if m == 0 and n == 0: continue
                idx = m * mosaic_shape[0] + n - 1
                if idx >= len(image_latents): 
                    idx = random.randint(0, len(image_latents) - 1)
                mosaic_image_latent[:,:,m*latent_height:(m+1)*latent_height,n*latent_width:(n+1)*latent_width] = image_latents[idx]
        # test_image = self.vae.decode(mosaic_image_latent / self.vae.config.scaling_factor + self.vae.config.shift_factor, return_dict=False)[0]
        # test_image = self.image_processor.postprocess(test_image, output_type='pil')
        # test_image[0].save("latent_mosaic_decode.png")
        # input()

        # print("mosaic_image_latent", mosaic_image_latent.shape) # torch.Size([1, 16, 192, 192])
        mosaic_latent_image_id = self._prepare_latent_image_ids(batch_size, mosaic_latent_height // 2, mosaic_latent_width // 2, device, dtype)

        mosaic_mask = torch.zeros((1, 1, mosaic_latent_height, mosaic_latent_width), device=image.device, dtype=image.dtype)
        mosaic_mask[:,:,:latent_height,:latent_width] = 1.0

        if batch_size > mosaic_image_latent.shape[0] and batch_size % mosaic_image_latent.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // mosaic_image_latent.shape[0]
                mosaic_image_latent = torch.cat([mosaic_image_latent] * additional_image_per_prompt, dim=0)
        elif batch_size > mosaic_image_latent.shape[0] and batch_size % mosaic_image_latent.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {mosaic_image_latent.shape[0]} to {batch_size} text prompts."
            )
        else:
            mosaic_image_latent = torch.cat([mosaic_image_latent], dim=0)
        # print(mosaic_image_latent.shape, mosaic_latent_image_id.shape, mosaic_mask.shape) # torch.Size([1, 16, 192, 192]) torch.Size([9216, 3]) torch.Size([1, 16, 192, 192])

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.scheduler.scale_noise(mosaic_image_latent, timestep, noise)
            # print(latents.shape) # torch.Size([1, 16, 192, 192]) 
        else:
            noise = latents.to(device)
            latents = noise

        noise = self._pack_latents(noise, batch_size, num_channels_latents, mosaic_latent_height, mosaic_latent_width)
        mosaic_image_latent = self._pack_latents(mosaic_image_latent, batch_size, num_channels_latents, mosaic_latent_height, mosaic_latent_width)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, mosaic_latent_height, mosaic_latent_width)
        mosaic_mask = self._pack_latents(mosaic_mask.repeat(1, num_channels_latents, 1, 1),
                batch_size,
                num_channels_latents,
                mosaic_latent_height,
                mosaic_latent_width,
            )
        # print(latents.shape, noise.shape, mosaic_image_latent.shape, mosaic_latent_image_id.shape, mosaic_mask.shape) # torch.Size([1, 9216, 64]) torch.Size([1, 9216, 64]) torch.Size([1, 9216, 64]) torch.Size([9216, 3]) torch.Size([1, 9216, 64])
        return latents, noise, mosaic_image_latent, mosaic_latent_image_id, mosaic_mask


def FluxTransformer2DModelForward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
    ##################################################
    height: int = None, step: int = 0,
    ##################################################
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    The [`FluxTransformer2DModel`] forward method.

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
    
    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]
    
    m = int(torch.max(img_ids[:, 1]).item()) + 1
    n = int(torch.max(img_ids[:, 2]).item()) + 1
    shifted_rotary_embs = []

    for s in self.cascade:
        id_mat = []
        for i in range(m // s):
            for j in range(n // s):
                id_mat.append([0., float(i), float(j)])
        id_mat_tensor = torch.tensor(id_mat, dtype=torch.float, device=img_ids.device)
        shift_ids = torch.cat((txt_ids, id_mat_tensor), dim=0)
        shift_image_rotary_emb = self.pos_embed(shift_ids)
        shifted_rotary_embs.append(shift_image_rotary_emb)

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

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
                image_rotary_emb,
                **ckpt_kwargs,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                ###############################################################################################################
                timestep=timestep, height=height // self.config.patch_size, shifted_rotary_embs=shifted_rotary_embs, step=step
                ###############################################################################################################
            )

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            # For Xlabs ControlNet.
            if controlnet_blocks_repeat:
                hidden_states = (
                    hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )

        else:
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                ###############################################################################################################
                timestep=timestep, height=height // self.config.patch_size, shifted_rotary_embs=shifted_rotary_embs, step=step
                ###############################################################################################################
            )

        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                + controlnet_single_block_samples[index_block // interval_control]
            )

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def FluxSingleTransformerBlockForward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ############################################################
    height: int = None,
    timestep: Optional[torch.Tensor] = None,
    shifted_rotary_embs: Optional[List] = None,
    step: int = 0,
    ############################################################
) -> torch.Tensor:
    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        ######################################################################################################
        timestep=timestep, height=height, shifted_rotary_embs=shifted_rotary_embs, step=step,
        ######################################################################################################
        **joint_attention_kwargs,
    )

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    gate = gate.unsqueeze(1)
    hidden_states = gate * self.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states


def FluxTransformerBlockForward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
    ############################################################
    height: int = None,
    timestep: Optional[torch.Tensor] = None,
    shifted_rotary_embs: Optional[List] = None,
    step: int = 0
    ############################################################
):
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )
    joint_attention_kwargs = joint_attention_kwargs or {}
    # Attention.
    attn_output, context_attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        ######################################################################################################
        timestep=timestep, height=height, shifted_rotary_embs=shifted_rotary_embs, step=step,
        ######################################################################################################
        **joint_attention_kwargs,
    )

    # Process attention outputs for the `hidden_states`.
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output

    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = hidden_states + ff_output

    # Process attention outputs for the `encoder_hidden_states`.

    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states


def get_qkv(hidden_states, attn, batch_size, head_dim):
    # print(hidden_states.shape) # torch.Size([1, 2048, 3072]) # torch.Size([1, 9216, 3072])

    # `sample` projections.
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)
    # print(query.shape) # torch.Size([1, 9216, 3072])

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)
    return query, key, value


def get_encoder_qkv(encoder_hidden_states, query, key, value, attn, batch_size, head_dim):
    # `context` projections for text tokens
    encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
    encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
    encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

    encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)
    encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)
    encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
        batch_size, -1, attn.heads, head_dim
    ).transpose(1, 2)

    if attn.norm_added_q is not None:
        encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
    if attn.norm_added_k is not None:
        encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

    # Concatenate text first, then image
    # text tokens = encoder_hidden_states_query_proj, image tokens = query
    query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
    key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
    value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
    return query, key, value



def pool_kq(k, q, kernel=(2, 2), stride=(2, 2), grid_shape=(3, 3), image_shape=(512, 512), text_length=512, is_single=False):
    mosaic_height = image_shape[0] // 16 * grid_shape[0]
    
    adjusted_k, adjusted_q = k.clone(), q.clone()
    if is_single:
        adjusted_k = adjusted_k[:, :, text_length:, :]
        adjusted_q = adjusted_q[:, :, text_length:, :]
    # Rearrange from (batch, attn_head, (height * width), feat_dim) -> (batch, attn_head, height, width, feat_dim)
    k_reshaped = rearrange(
        adjusted_k, 'b d (h w) f -> b d h w f',
        h=mosaic_height
    ) 
    q_reshaped = rearrange(
        adjusted_q, 'b d (h w) f -> b d h w f',
        h=mosaic_height
    ) 

    # Temporarily reshape to (batch * attn_head, feat_dim, height, width)
    k_reshaped = k_reshaped.permute(0, 1, 4, 2, 3).contiguous()  # (batch, attn_head, feat_dim, height, width)
    q_reshaped = q_reshaped.permute(0, 1, 4, 2, 3).contiguous()

    batch, attn_head, feat_dim, height, width = k_reshaped.shape

    # Reshape to (batch * attn_head * feat_dim, 1, height, width) for pooling
    k_reshaped = k_reshaped.view(batch * attn_head * feat_dim, 1, height, width)
    q_reshaped = q_reshaped.view(batch * attn_head * feat_dim, 1, height, width)

    # Apply pooling
    k_pooled = F.avg_pool2d(k_reshaped, kernel_size=kernel, stride=stride)
    q_pooled = F.avg_pool2d(q_reshaped, kernel_size=kernel, stride=stride)

    # Compute new height & width
    new_height = k_pooled.shape[-2]
    new_width = k_pooled.shape[-1]

    # Reshape back to (batch, attn_head, new_height, new_width, feat_dim)
    k_pooled = k_pooled.view(batch, attn_head, feat_dim, new_height, new_width).permute(0, 1, 3, 4, 2)
    q_pooled = q_pooled.view(batch, attn_head, feat_dim, new_height, new_width).permute(0, 1, 3, 4, 2)

    # Flatten spatial dimensions
    k_pooled = rearrange(k_pooled, 'b d h w f -> b d (h w) f')
    q_pooled = rearrange(q_pooled, 'b d h w f -> b d (h w) f')

    if is_single:
        k_pooled = torch.cat([k[:, :, :text_length, :], k_pooled], dim=2)
        q_pooled = torch.cat([q[:, :, :text_length, :], q_pooled], dim=2)
    return k_pooled, q_pooled


# FluxAttnProcessor2_0
def flux_attn_call2_0(
    self,
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    ############################################################
    height: int = None,
    timestep: Optional[torch.Tensor] = None,
    shifted_rotary_embs: Optional[List] = None,
    step: int = 0
    ############################################################
) -> torch.FloatTensor:

    batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

    inner_dim = hidden_states.shape[-1]
    head_dim = inner_dim // attn.heads

    query, key, value = get_qkv(hidden_states, attn, batch_size, head_dim)
    # print(key.shape) # torch.Size([1, 24, 9216, 128])

    cascade_scales = self.cascade 
    cascaded_qks = []
    for s in cascade_scales: 
        pooled_query, pooled_key = pool_kq(key, query, kernel=(s, s), stride=(s, s), grid_shape=self.grid_shape, image_shape=self.image_shape, is_single=encoder_hidden_states is None)
        cascaded_qks.append((pooled_query, pooled_key))

    # Handle cross-attention (text + image)
    if encoder_hidden_states is not None:
        query, key, value = get_encoder_qkv(encoder_hidden_states, query, key, value, attn, batch_size, head_dim)
        # print(query.shape) # torch.Size([1, 24, 9728, 128])
        cascaded_encoder_qks = []
        for q, k in cascaded_qks: 
            qq, kk, _ = get_encoder_qkv(encoder_hidden_states, q, k, value, attn, batch_size, head_dim)
            cascaded_encoder_qks.append((qq, kk))
        cascaded_qks = cascaded_encoder_qks

    cascaded_sqks_rot = []
    if image_rotary_emb is not None: # 1024x1024 (torch.Size([4608, 128]) torch.Size([4608, 128]))
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

        for i, sre in enumerate(shifted_rotary_embs):
            q, k = cascaded_qks[i]
            cascaded_sqks_rot.append((cascade_scales[i], apply_rotary_emb(q, sre), apply_rotary_emb(k, sre)))

    # Compute attention
    if hasattr(self, "aug_att") and self.aug_att != 0.0 and encoder_hidden_states is not None and step < self.steps: 
        hidden_states, _ = scaled_dot_product_attention_cascade(
            query, key, value, 
            dropout_p=0.0, aug_att=self.aug_att, grid_shape=self.grid_shape, 
            image_shape=self.image_shape, cascaded_sqks_rot=cascaded_sqks_rot
        )
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    if encoder_hidden_states is not None:
        # Split back into encoder (text) and image parts
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
    else:
        return hidden_states


def get_attention_prob(query, key, scale=None, need_softmax=True):
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if need_softmax:
        attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight


def scaled_dot_product_attention_cascade(query, key, value, dropout_p=0.0, scale=None, aug_att=0.0, grid_shape=(3,3), image_shape=(512,512), cascaded_sqks_rot=None, text_len=512) -> torch.Tensor:
    attn_weight = get_attention_prob(query, key, scale=scale, need_softmax=True)
    
    height, width = image_shape[0] // 16, image_shape[1] // 16 
    mosaic_height = height * grid_shape[0]
    attention_probs_image = attn_weight[:, :, text_len:, text_len:]
    attention_probs_reshape = rearrange(
            attention_probs_image,
            'b d (h1 w1) (h2 w2) -> b d h1 w1 h2 w2',
            h1=mosaic_height, 
            h2=mosaic_height,
        ) 
    c_scalor = aug_att / len(cascaded_sqks_rot)
    for s, q, k in cascaded_sqks_rot:
        ajusted_c_scalor = c_scalor/s
        c_attn_weight = get_attention_prob(q, k, scale=scale, need_softmax=True)

        c_attn_weight_image = c_attn_weight[:, :, text_len:, text_len:]
        c_attn_weight_image_upsampled = F.interpolate(
                                                c_attn_weight_image, 
                                                size=attention_probs_image.shape[-2:], 
                                                mode="bilinear", 
                                                align_corners=False
                                            )
        # print(s, c_attn_weight_image_upsampled.shape, attention_probs_image.shape)
        c_attn_weight_probs_reshape = rearrange(
            c_attn_weight_image_upsampled,
            'b d (h1 w1) (h2 w2) -> b d h1 w1 h2 w2',
            h1=mosaic_height, 
            h2=mosaic_height,
        ) 
        attention_probs_reshape[:,:,:height,:width,:height,width:] += c_attn_weight_probs_reshape[:,:,:height,:width,:height,width:] * ajusted_c_scalor
        attention_probs_reshape[:,:,:height,:width,height:,:] += c_attn_weight_probs_reshape[:,:,:height,:width,height:,:] * ajusted_c_scalor

    return torch.dropout(attn_weight, dropout_p, train=False) @ value, attn_weight

