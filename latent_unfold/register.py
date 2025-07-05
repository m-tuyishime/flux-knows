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
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import numpy as np
import torch.nn as nn
import types
from safetensors.torch import load_file
import tqdm

from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from diffusers.models.attention_processor import FluxAttnProcessor2_0

from .latent_unfold import *


# for visualization
def hook_function(name, detach=True):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            timestep = getattr(module.processor, "timestep", None)
            attn_map = getattr(module.processor, "attn_map", None)

            if timestep is not None and attn_map is not None:
                attn_maps[timestep] = attn_maps.get(timestep, dict())
                attn_maps[timestep][name] = attn_map.cpu() if detach else attn_map
            
            # Clean up
            if hasattr(module.processor, "attn_map"):
                del module.processor.attn_map

    return forward_hook


def register_attention_hook(model, hook_function, target_name, aug_att=0.0, grid_shape=(3,3), image_shape=(512, 512), cascade=(2,3,4), steps=10):
    for name, module in model.named_modules():
        if not name.endswith(target_name):
            continue
        elif isinstance(module.processor, FluxAttnProcessor2_0):
            # print("FluxAttnProcessor2_0")
            module.processor.aug_att = aug_att if aug_att > 0.0 else 0.0
            module.processor.grid_shape = grid_shape
            module.processor.image_shape = image_shape
            module.processor.cascade = cascade
            module.processor.steps = steps
        hook = module.register_forward_hook(hook_function(name))

    return model


def replace_call_method_for_flux(model, cascade=(2,3,4)):
    if model.__class__.__name__ == 'FluxTransformer2DModel':
        model.forward = FluxTransformer2DModelForward.__get__(model, FluxTransformer2DModel)
        model.cascade = cascade

    for name, layer in model.named_children():
        if layer.__class__.__name__ == 'FluxTransformerBlock':
            layer.forward = FluxTransformerBlockForward.__get__(layer, FluxTransformerBlock)
        if layer.__class__.__name__ == 'FluxSingleTransformerBlock':
            layer.forward = FluxSingleTransformerBlockForward.__get__(layer, FluxSingleTransformerBlock)
        replace_call_method_for_flux(layer)
    
    return model


def init_pipeline(pipeline, aug_att=0.0, grid_shape=(3,3), image_shape=(512,512), cascade=(2,3,4), steps=10):
    FluxAttnProcessor2_0.__call__ = flux_attn_call2_0

    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
            pipeline.transformer = register_attention_hook(pipeline.transformer, hook_function, 'attn', aug_att=aug_att, grid_shape=grid_shape, image_shape=image_shape, cascade=cascade, steps=steps)
            pipeline.transformer = replace_call_method_for_flux(pipeline.transformer, cascade=cascade)
            
    return pipeline





