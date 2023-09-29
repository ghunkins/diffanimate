#
#    ___  _______________   _  ________  ______ __________
#   / _ \/  _/ __/ __/ _ | / |/ /  _/  |/  / _ /_  __/ __/
#  / // // // _// _// __ |/    // // /|_/ / __ |/ / / _/  
# /____/___/_/ /_/ /_/ |_/_/|_/___/_/  /_/_/ |_/_/ /___/  
#     
import math
from typing import Optional
from diffusers.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from diffusers import TextToVideoSDPipeline

import torch
import PIL.Image
import numpy as np


class DiffAnimatePipeline(TextToVideoSDPipeline):
    """
    Pipeline for DiffAnimate generations.
    """

    def prepare_latents(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):
    
        # if image is None, use super
        image = getattr(self, 'image')
        if image is None:
            return super().prepare_latents(
                batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents
            )

        # otherwise
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if image is not None:
            if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
                raise ValueError(
                    f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
                )
            image = image.to(device=device, dtype=dtype)
            if isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)
        else:
            init_latents = None


        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
                if init_latents is not None:
                    for i in range(num_frames):
                        # TODO (@ghunkins): figure out if this is the optimal
                        # decay function
                        decay = ((math.exp(-i / num_frames) ** 6) * 0.7) + 0.3
                        init_alpha = (1 / 20) * decay * (1 - self.strength)
                        latents[:, :, i, :, :] = init_latents * init_alpha + latents[:, :, i, :, :] * (1 - init_alpha)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if init_latents is None:
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def postprocess_mask(self, result: 'TextToVideoSDPipelineOutput') -> 'TextToVideoSDPipelineOutput':
        if self.mask_image is None:
            return result
        masked_frames = []
        for frame in result.frames:
            output = PIL.Image.fromarray(frame)
            masked_frame = PIL.Image.composite(
                output,
                self.origin_image.resize(output.size),
                self.mask_image.filter(PIL.ImageFilter.GaussianBlur(3))
                .resize(output.size)
                .convert("L"),
            )
            masked_frames.append(np.array(masked_frame))

        result.frames = masked_frames
        return result
        

    def __call__(self, *args, **kwargs):
        if 'image' in kwargs:
            image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
            self.origin_image = kwargs.pop('image')
            self.image = image_processor.preprocess(self.origin_image)
            self.strength = kwargs.pop('strength', 0.5)
            if 'mask_image' in kwargs:
                self.mask_image = kwargs.pop('mask_image')
            else:
                self.mask_image = None

            kwargs['width'] = self.origin_image.width
            kwargs['height'] = self.origin_image.height
        else:
            self.origin_image = None
            self.image = None
            self.mask_image = None

        result = super().__call__(*args, **kwargs)
        result = self.postprocess_mask(result)
        return result
