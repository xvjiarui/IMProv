# Modified from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
# by Jiarui Xu

# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import os
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
import PIL
import torch
import torch.utils.checkpoint
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import logging

from improv.modeling.meta_arch.modules import (
    PatchEmbedEncoder,
    TransformerMAE,
    VQDecoder,
    VQEncoder,
)

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def mask_by_random_topk(probs, k, temperature=1.0):
    device = probs.device
    dtype = probs.dtype
    gumbel_noise = torch.distributions.Gumbel(
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.tensor(1.0, device=device, dtype=dtype),
    ).sample(probs.shape)
    confidence = torch.log(probs) + temperature * gumbel_noise
    # Obtains cut off threshold given the mask lengths.
    cut_off = confidence.topk(k, dim=-1).values[:, -1]
    # Masks tokens with lower confidence.
    # [batch_size, seq_len]
    masking = confidence < cut_off
    return masking


class MAEScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        schedule: str = "cosine",
    ):
        if schedule == "cosine":
            self.schedule_fn = lambda t: torch.cos(t * np.pi * 0.5)
        elif schedule.startswith("const"):
            mask_ratio = 0.75  # by default
            if schedule.split("const")[-1].isnumeric():
                mask_ratio = float(schedule.split("const")[-1]) / 100
            self.schedule_fn = lambda t: mask_ratio * torch.ones_like(t)
        else:
            raise ValueError(f"Schedule {schedule} not supported.")

    @staticmethod
    def rand_masking_token(x, mask_ratio):
        N, L = x.shape[:2]  # batch, length

        len_keep = int(L * (1 - mask_ratio))

        noise = randn_tensor((N, L), device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        if x.ndim == 3:
            D = x.shape[-1]  # dimension
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        else:
            x_masked = torch.gather(x, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device, dtype=x.dtype)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    @staticmethod
    def det_masking_token(x, input_mask):
        N, L = x.shape[:2]  # batch, length
        assert input_mask.shape == (N, L), input_mask.shape

        len_keep = (L * (1 - input_mask.mean(dim=1))).int()
        # assert torch.all(len_keep == len_keep[0]), len_keep
        if not torch.all(len_keep == len_keep[0]):
            logger.warning("masking length is not the same for all samples")
            logger.warning(len_keep)
        len_keep = len_keep[0].item()

        # sort noise for each sample
        ids_shuffle = torch.argsort(input_mask, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        if x.ndim == 3:
            D = x.shape[-1]  # dimension
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        else:
            x_masked = torch.gather(x, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device, dtype=input_mask.dtype)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # assert torch.allclose(mask, input_mask)

        return x_masked, mask, ids_restore

    @staticmethod
    def det_masking_token_min(x, input_mask):
        N, L = x.shape[:2]  # batch, length
        assert input_mask.shape == (N, L), input_mask.shape

        len_keep = (L * (1 - input_mask.mean(dim=1))).int()
        len_keep = len_keep.min().item()

        # sort noise for each sample
        ids_shuffle = torch.argsort(input_mask, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        if x.ndim == 3:
            D = x.shape[-1]  # dimension
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        else:
            x_masked = torch.gather(x, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device, dtype=input_mask.dtype)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


def prepare_mask_and_masked_image(image, mask):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """
    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert (
            image.shape[-2:] == mask.shape[-2:]
        ), "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 255.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


class IMProvPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vqvae ([`VQModel`]):
            Vector-quantized (VQ) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """
    _optional_components = ["text_encoder", "tokenizer"]

    def __init__(
        self,
        image_encoder: Union[PatchEmbedEncoder, VQEncoder],
        image_decoder: VQDecoder,
        scheduler: MAEScheduler,
        mask_image_model: TransformerMAE,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()
        self.register_modules(
            image_encoder=image_encoder,
            image_decoder=image_decoder,
            scheduler=scheduler,
            mask_image_model=mask_image_model,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        from diffusers.pipelines.pipeline_utils import LOADABLE_CLASSES
        for new_lib in ['improv.pipelines.pipeline_improv', 'improv.modeling.meta_arch.modules']:
            if new_lib not in LOADABLE_CLASSES:
                LOADABLE_CLASSES[new_lib] = {
                    "ModelMixin": ["save_pretrained", "from_pretrained"],
                    "SchedulerMixin": ["save_pretrained", "from_pretrained"],
                }
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def extract_init_dict(cls, config_dict, **kwargs):
        from diffusers.pipelines.pipeline_utils import LOADABLE_CLASSES

        init_dict, unused_kwargs, hidden_config_dict = super().extract_init_dict(
            config_dict, **kwargs
        )
        for k in init_dict:
            lib, mod = init_dict[k]
            new_lib = None
            if mod in ["PatchEmbedEncoder", "TransformerMAE", "VQDecoder", "VQEncoder"]:
                new_lib = "improv.modeling.meta_arch.modules"
            elif mod in ["MAEScheduler"]:
                new_lib = "improv.pipelines.pipeline_improv"
            # if lib is None or mod is None:
            #     init_dict[k] = None
            if new_lib is not None:
                init_dict[k] = (new_lib, mod)
                if new_lib not in LOADABLE_CLASSES:
                    LOADABLE_CLASSES[new_lib] = {
                        "ModelMixin": ["save_pretrained", "from_pretrained"],
                        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
                    }

        return init_dict, unused_kwargs, hidden_config_dict

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if self.text_encoder is None:
            return None
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            # untruncated_ids = self.tokenizer(
            #     prompt, padding="longest", return_tensors="pt"
            # ).input_ids

            # if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            #     text_input_ids, untruncated_ids
            # ):
            #     removed_text = self.tokenizer.batch_decode(
            #         untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            #     )
            #     logger.warning(
            #         "The following part of your input was truncated because CLIP can only handle sequences up to"
            #         f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            #     )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ) or "T5" in self.text_encoder.config.architectures[0]:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None and self.text_encoder is not None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 * C)
        """
        expanded_dim = False
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(1)
            expanded_dim = True
        # set to 1 for now
        p = 1
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        c = imgs.shape[1]
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        if expanded_dim:
            x = x.squeeze(-1)
        return x

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 5,
        guidance_scale: float = 7.5,
        categorical_sampling: bool = False,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        choice_temperature: float = 4.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        image_tokens: Optional[torch.LongTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
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
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = (
            height
            or self.mask_image_model.config.sample_size * self.image_decoder.config.scale_factor
        )
        width = (
            width
            or self.mask_image_model.config.sample_size * self.image_decoder.config.scale_factor
        )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif self.text_encoder is None:
            batch_size = len(image)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 and self.text_encoder is not None

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Preprocess mask and image
        if mask_image is not None:
            mask, masked_image = prepare_mask_and_masked_image(image, mask_image)
            # TODO: use masked image or not
            image_latents = self.patchify(self.image_encoder(masked_image))
            # downsampling mask, nearest will loss boarder
            mask = torch.nn.functional.interpolate(
                mask,
                size=(
                    height // self.image_encoder.scale_factor,
                    width // self.image_encoder.scale_factor,
                ),
                mode="bilinear",
            )
            mask = torch.where(mask > 0.0, torch.ones_like(mask), torch.zeros_like(mask))
            mask = self.patchify(mask).squeeze(-1)
        else:
            image_latents = self.patchify(self.image_encoder(image))
            mask = torch.ones(
                (
                    batch_size,
                    height // self.image_encoder.scale_factor,
                    width // self.image_encoder.scale_factor,
                ),
                dtype=image_tokens.dtype,
                device=device,
            )
            mask = self.patchify(mask).squeeze(-1)
        masked_image_latents, _, ids_restore = self.scheduler.det_masking_token(image_latents, mask)

        # image_tokens = self.mask_image_model(
        #     masked_image_latents,
        #     ids_restore=ids_restore,
        #     encoder_hidden_states=prompt_embeds[-batch_size:] if prompt_embeds is not None else None,
        # ).argmax(dim=-1)
        image_tokens = None
        unknown_map = mask > 0.5

        assert mask[0].allclose(mask.mean(dim=0)), "mask should be same for all images in batch"

        timesteps = torch.linspace(0, 1, num_inference_steps, device=device)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                token_model_input = (
                    torch.cat([masked_image_latents] * 2)
                    if do_classifier_free_guidance
                    else masked_image_latents
                )
                ids_restore = (
                    torch.cat([ids_restore] * 2) if do_classifier_free_guidance else ids_restore
                )

                logits = self.mask_image_model(
                    token_model_input,
                    ids_restore=ids_restore,
                    encoder_hidden_states=prompt_embeds,
                )

                # perform guidance
                if do_classifier_free_guidance:
                    logits_uncond, logits = logits.chunk(2)
                    time_guidance_scale = 1 + t * (guidance_scale - 1)
                    logits = logits_uncond + time_guidance_scale * (logits - logits_uncond)

                # [B, L]
                # TODO: check shape, check need norm or not
                # Samples the ids using categorical sampling: [batch_size, seq_length].
                if categorical_sampling:
                    sampled_image_tokens = torch.distributions.Categorical(logits=logits).sample()
                else:
                    sampled_image_tokens = logits.argmax(dim=-1)

                if image_tokens is None:
                    image_tokens = sampled_image_tokens
                else:
                    # Just updates the masked tokens.
                    image_tokens = torch.where(unknown_map, sampled_image_tokens, image_tokens)

                sampled_image_latents = self.patchify(
                    self.image_encoder(
                        self.image_decoder(
                            sampled_image_tokens.view(
                                batch_size,
                                height // self.image_decoder.scale_factor,
                                width // self.image_decoder.scale_factor,
                            )
                        )
                    )
                )
                image_latents = torch.where(
                    unknown_map.unsqueeze(-1), sampled_image_latents, image_latents
                )

                mask_ratio = self.scheduler.schedule_fn(t).item()

                # [B, L, C]
                probs = torch.log_softmax(logits, dim=-1)
                # [B, L]
                selected_probs = probs.gather(-1, sampled_image_tokens.unsqueeze(-1)).squeeze(-1)
                selected_probs = torch.where(unknown_map, selected_probs, float("inf"))

                gumbel_noise = torch.distributions.Gumbel(
                    torch.tensor(0.0, device=device, dtype=selected_probs.dtype),
                    torch.tensor(1.0, device=device, dtype=selected_probs.dtype),
                ).sample(selected_probs.shape)
                # [B, L]
                noise_confidence = selected_probs + gumbel_noise * choice_temperature * (1.0 - t)
                # calculate the cut-off w.r.t initial mask
                # TODO: leaved for bar here
                cut_off_len = (mask_ratio * mask.sum(dim=1)).int()
                # cut_off_len = (mask_ratio * torch.ones_like(mask).sum(dim=1)).int()
                # Keeps at least one of prediction in this round and also masks out at least
                # one and for the next iteration
                cut_off_len = torch.maximum(
                    torch.ones_like(cut_off_len),
                    torch.minimum(torch.sum(unknown_map, dim=-1) - 1, cut_off_len),
                )
                # NOTE: comment out of DDL
                # assert all(
                #     cut_off_len[0].allclose(cut_off_len[i]) for i in range(batch_size)
                # ), "cut_off_len should be same for all images in batch"
                cut_off_thresh = cut_off_len[0].item() / mask.shape[1] + 1e-6

                cut_off = torch.quantile(noise_confidence, cut_off_thresh, dim=1, keepdim=True)
                unknown_map &= noise_confidence < cut_off

                masked_image_latents, _, ids_restore = self.scheduler.det_masking_token_min(
                    image_latents, unknown_map.to(logits.dtype)
                )

                # call the callback, if provided
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, image_tokens)

        image_tokens = image_tokens.view(
            batch_size,
            height // self.image_decoder.scale_factor,
            width // self.image_decoder.scale_factor,
        )

        if output_type == "token":
            image = image_tokens
        elif output_type == "pil":
            # 8. Post-processing
            image = self.image_decoder(image_tokens).cpu().permute(0, 2, 3, 1).float().numpy()
            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        elif output_type == "numpy":
            # 8. Post-processing
            image = self.image_decoder(image_tokens).cpu().permute(0, 2, 3, 1).float().numpy()
        elif output_type == "torch":
            # 8. Post-processing
            image = self.image_decoder(image_tokens)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
