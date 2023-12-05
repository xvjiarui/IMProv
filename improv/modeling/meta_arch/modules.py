import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusers import VQModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import get_2d_sincos_pos_embed
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformer_2d import BasicTransformerBlock

logger = logging.getLogger(__name__)


def set_norm_eps(module, eps=1e-6, norm_type="LN"):
    if norm_type == "LN":
        norm_cls = nn.LayerNorm
    else:
        raise NotImplementedError
    for m in module.modules():
        if isinstance(m, norm_cls):
            assert hasattr(m, "eps"), f"{m} does not have eps attribute"
            m.eps = eps


class VQEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        pixel_mean: Tuple[float] = (0.5, 0.5, 0.5),
        pixel_std: Tuple[float] = (0.5, 0.5, 0.5),
    ):
        super().__init__()
        vq_model = VQModel.from_pretrained(pretrained_model_name_or_path)
        self.encoder = vq_model.encoder
        self.quant_conv = vq_model.quant_conv
        self.quantize = vq_model.quantize

        # del self.pixel_mean, self.pixel_std
        self.register_buffer("pt_pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pt_pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.register_to_config(scale_factor=2 ** (len(vq_model.config.block_out_channels) - 1))
        self.register_to_config(codebook_size=vq_model.config.num_vq_embeddings)

    def forward(self, x):
        x = (x - self.pt_pixel_mean) / self.pt_pixel_std

        x = self.encoder(x)
        x = self.quant_conv(x)
        latent, _, (_, _, token_ids) = self.quantize(x)
        # [B, H, W]
        token_ids = token_ids.reshape(latent.shape[0], *latent.shape[-2:])

        return token_ids


class VQDecoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        pixel_mean: Tuple[float] = (0.5, 0.5, 0.5),
        pixel_std: Tuple[float] = (0.5, 0.5, 0.5),
    ):
        super().__init__()
        vq_model = VQModel.from_pretrained(pretrained_model_name_or_path)
        self.quantize = vq_model.quantize
        self.post_quant_conv = vq_model.post_quant_conv
        self.decoder = vq_model.decoder

        # del self.pixel_mean, self.pixel_std
        self.register_buffer("pt_pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pt_pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        self.register_to_config(scale_factor=2 ** (len(vq_model.config.block_out_channels) - 1))
        self.register_to_config(codebook_size=vq_model.config.num_vq_embeddings)

    def forward(self, token_ids):
        # [N, C. H, W]
        y = self.quantize.get_codebook_entry(token_ids, shape=None).permute(0, 3, 1, 2)
        y = self.post_quant_conv(y)
        y = self.decoder(y)

        y = y * self.pt_pixel_std + self.pt_pixel_mean

        y = torch.clamp(y, 0, 1)

        return y


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PositionEmbedding2D(nn.Module):
    def __init__(self, grid_size, embed_dim):
        super().__init__()
        self.grid_size = grid_size
        pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)
        pos_embed = rearrange(pos_embed, "(h w) c -> 1 c h w", h=grid_size, w=grid_size)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float(), persistent=False)

    def forward(self, x):
        if x.ndim == 4:
            pos_embed = F.interpolate(
                self.pos_embed, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            x = x + pos_embed
        elif x.ndim == 3:
            grid_size = int(x.shape[1] ** 0.5)
            pos_embed = F.interpolate(
                self.pos_embed, size=(grid_size, grid_size), mode="bilinear", align_corners=False
            )
            pos_embed = rearrange(pos_embed, "b c h w -> b (h w) c")
            x = x + pos_embed

        return x


class PatchEmbedEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        bias=True,
        pixel_mean: Tuple[float] = (0.5, 0.5, 0.5),
        pixel_std: Tuple[float] = (0.5, 0.5, 0.5),
    ):
        super().__init__()
        num_patches = (height // patch_size) * (width // patch_size)
        self.layer_norm = layer_norm

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=bias,
        )
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if layer_norm:
            self.norm = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        else:
            self.norm = None

        self.pos_embed = PositionEmbedding2D(int(num_patches**0.5), embed_dim)

        # del self.pixel_mean, self.pixel_std
        self.register_buffer("pt_pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pt_pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        self.register_to_config(scale_factor=patch_size)

    def forward(self, x):

        x = (x - self.pt_pixel_mean) / self.pt_pixel_std

        x = self.proj(x)
        if self.layer_norm:
            x = self.norm(x)
        x = self.pos_embed(x)

        return x


class TransformerMAE(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        out_channels: int,
        in_channels: Optional[int] = None,
        num_embed: Optional[int] = None,
        encoder_embed_dim: int = 1024,
        encoder_depth: int = 24,
        encoder_num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        encoder_cross_attention: bool = True,
        decoder_cross_attention: bool = True,
        cross_attention_dim: Optional[int] = None,
        with_cls_token: bool = False,
        sample_size: int = 14,
        in_context_cross_attention: bool = False,
        encoder_cross_attention_cat_encoder: bool = False,
        decoder_cross_attention_cat_encoder: bool = False,
    ):

        super().__init__()
        assert (in_channels is None) != (
            num_embed is None
        ), "Only one of in_channels or num_embed can be specified"

        if in_channels is not None:
            if in_channels == encoder_embed_dim:
                self.emb = nn.Identity()
            else:
                self.emb = nn.Linear(in_channels, encoder_embed_dim)

        if num_embed is not None:
            self.emb = nn.Embedding(num_embed, encoder_embed_dim)

        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.) # noqa
            torch.nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        if in_context_cross_attention:
            transformer_cls = BasicTransformerBlockInContext
        else:
            transformer_cls = BasicTransformerBlock

        self.encoder_blocks = nn.ModuleList(
            [
                transformer_cls(
                    dim=encoder_embed_dim,
                    num_attention_heads=encoder_num_heads,
                    attention_head_dim=encoder_embed_dim // encoder_num_heads,
                    dropout=0.0,
                    cross_attention_dim=cross_attention_dim if encoder_cross_attention else None,
                    activation_fn="gelu",
                    num_embeds_ada_norm=None,
                    attention_bias=True,
                    only_cross_attention=False,
                    upcast_attention=False,
                    norm_elementwise_affine=True,
                    norm_type="layer_norm",
                    final_dropout=False,
                )
                for _ in range(encoder_depth)
            ]
        )
        set_norm_eps(self.encoder_blocks, 1e-6)
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.decoder_pos_embed = PositionEmbedding2D(sample_size, decoder_embed_dim)

        self.decoder_blocks = nn.ModuleList(
            [
                transformer_cls(
                    dim=decoder_embed_dim,
                    num_attention_heads=decoder_num_heads,
                    attention_head_dim=decoder_embed_dim // decoder_num_heads,
                    dropout=0.0,
                    cross_attention_dim=cross_attention_dim if decoder_cross_attention else None,
                    activation_fn="gelu",
                    num_embeds_ada_norm=None,
                    attention_bias=True,
                    only_cross_attention=False,
                    upcast_attention=False,
                    norm_elementwise_affine=True,
                    norm_type="layer_norm",
                    final_dropout=False,
                )
                for _ in range(decoder_depth)
            ]
        )
        set_norm_eps(self.decoder_blocks, 1e-6)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, out_channels, bias=True
        )  # decoder to patch

        if encoder_cross_attention_cat_encoder or decoder_cross_attention_cat_encoder:
            self.encoder_cross_attn_proj = nn.Linear(
                encoder_embed_dim, cross_attention_dim, bias=True
            )
        self.encoder_cross_attention_cat_encoder = encoder_cross_attention_cat_encoder
        self.decoder_cross_attention_cat_encoder = decoder_cross_attention_cat_encoder

    def forward(self, x, ids_restore, encoder_hidden_states=None):

        x = self.emb(x)
        with_cls_token = self.cls_token is not None
        # append cls token
        if with_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.encoder_blocks:
            if self.encoder_cross_attention_cat_encoder:
                x = blk(
                    x,
                    encoder_hidden_states=torch.cat(
                        [encoder_hidden_states, self.encoder_cross_attn_proj(x)], dim=1
                    ),
                )
            else:
                x = blk(x, encoder_hidden_states=encoder_hidden_states)
        x = self.encoder_norm(x)

        if self.decoder_cross_attention_cat_encoder:
            encoder_cross_attn_x = self.encoder_cross_attn_proj(x)
            encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_cross_attn_x], dim=1)

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        cls_token_offset = 1 if with_cls_token else 0
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + cls_token_offset - x.shape[1], 1
        )
        x_ = torch.cat([x[:, cls_token_offset:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle

        # we assume the position encoding of [CLS] token is zero
        x_ = self.decoder_pos_embed(x_)

        x = torch.cat([x[:, :cls_token_offset, :], x_], dim=1)  # append cls token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, encoder_hidden_states=encoder_hidden_states)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, cls_token_offset:, :]

        return x


class BasicTransformerBlockInContext(BasicTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
    ):
        super().__init__(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            attention_bias=attention_bias,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_type=norm_type,
            final_dropout=final_dropout,
        )
        del self.attn2, self.norm2
        self.encoder_projection = nn.Linear(cross_attention_dim, dim, bias=False)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Self-Attention
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        if encoder_hidden_states is not None:
            encoder_hidden_states = self.encoder_projection(encoder_hidden_states)
            norm_hidden_states = torch.cat((norm_hidden_states, encoder_hidden_states), dim=1)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        attn_output = attn_output[:, : hidden_states.shape[1]]
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states
