# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, Tuple, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F


def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    """
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * torch.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class SingleLayerTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
    ):
        """
        Single layer transformer decoder for segmentation following the architecture 
        from the provided files and image.
        
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # Single transformer decoder layer components (as shown in the image)
        self.num_heads = nheads
        
        # Cross-attention layer (Masked Attention in the image)
        self.cross_attention = CrossAttentionLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dropout=0.0,
            normalize_before=pre_norm,
        )
        
        # Self-attention layer
        self.self_attention = SelfAttentionLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dropout=0.0,
            normalize_before=pre_norm,
        )
        
        # FFN layer
        self.ffn = FFNLayer(
            d_model=hidden_dim,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            normalize_before=pre_norm,
        )

        self.post_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query positional encoding
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        # Initial prediction before the single transformer layer
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )

        # Single transformer decoder layer (as shown in the architecture image)
        # 1. Cross-attention (Masked Attention with image features)
        level_index = 0  # Use the first level for simplicity, can be made configurable
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
        
        output = self.cross_attention(
            output,
            src[level_index],
            memory_mask=attn_mask,
            memory_key_padding_mask=None,
            pos=pos[level_index],
            query_pos=query_embed,
        )

        # 2. Self-attention
        output = self.self_attention(
            output, 
            tgt_mask=None, 
            tgt_key_padding_mask=None, 
            query_pos=query_embed
        )

        # 3. FFN
        output = self.ffn(output)

        # Final prediction after the single transformer layer
        outputs_class, outputs_mask, _ = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )

        out = {
            "pred_logits": outputs_class,
            "pred_masks": outputs_mask,
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.post_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (
            attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask


# Simplified Pixel Decoder (keeping the interface but simplified for this example)
class SimplePixelDecoder(nn.Module):
    def __init__(
        self,
        input_shape: Dict[str, Tuple[int]],
        conv_dim: int = 256,
        mask_dim: int = 256,
    ):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1][-1])
        self.in_features = [k for k, _ in input_shape]
        self.feature_channels = [v[0] for k, v in input_shape]
        
        # Simple 1x1 conv to project features
        self.proj_conv = Conv2d(self.feature_channels[-1], conv_dim, kernel_size=1)
        c2_xavier_fill(self.proj_conv)
        
        # Mask feature projection
        self.mask_feature = Conv2d(conv_dim, mask_dim, kernel_size=1)
        c2_xavier_fill(self.mask_feature)

    def forward_features(self, features):
        # Use the highest resolution feature
        x = features[self.in_features[-1]]
        projected = self.proj_conv(x)
        mask_features = self.mask_feature(projected)
        
        # Create multi-scale features (simplified - just use the same feature at different scales)
        multi_scale_features = [projected, projected, projected]
        
        return mask_features, projected, multi_scale_features


class SingleLayerSegmentationHead(nn.Module):
    def __init__(
        self,
        input_shape: Dict[str, Tuple[int]],  # ShapeSpec: [channels, height, width, stride]
        hidden_dim: int = 256,
        num_classes: int = 919,  # ImageNet-S has 919 semantic classes
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        transformer_in_feature: str = "multi_scale_pixel_decoder",
    ):
        """
        Single layer segmentation head following the architecture in the provided files and image.
        
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            hidden_dim: transformer hidden dimension
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        orig_input_shape = input_shape
        input_shape = sorted(input_shape.items(), key=lambda x: x[1][-1])
        self.in_features = [k for k, _ in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        # Simplified pixel decoder
        self.pixel_decoder = SimplePixelDecoder(
            input_shape=orig_input_shape,
            conv_dim=hidden_dim,
            mask_dim=hidden_dim,
        )
        
        # Single layer transformer decoder
        self.predictor = SingleLayerTransformerDecoder(
            in_channels=hidden_dim,
            mask_classification=True,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            pre_norm=False,
            mask_dim=hidden_dim,
            enforce_input_project=False,
        )

        self.transformer_in_feature = transformer_in_feature
        self.num_classes = num_classes

    def forward_features(self, features, mask=None):
        return self.layers(features, mask)

    def forward(self, features, mask=None):
        output = self.forward_features(features, mask)
        return output

    def predict(self, features, mask=None, rescale_to=(224, 224)):
        output = self.forward_features(features, mask)
        output["pred_masks"] = F.interpolate(
            output["pred_masks"],
            size=rescale_to,
            mode="bilinear",
            align_corners=False,
        )
        return output

    def layers(self, features, mask=None):
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        return predictions


# Example usage:
if __name__ == "__main__":
    # Input shapes for 224x224 resolution with typical backbone (e.g., ResNet-50)
    input_shape = {
        "res2": (256, 56, 56, 4),     # (channels, height, width, stride) - 224/4 = 56
        "res3": (512, 28, 28, 8),     # 224/8 = 28  
        "res4": (1024, 14, 14, 16),   # 224/16 = 14
        "res5": (2048, 7, 7, 32),     # 224/32 = 7
    }
    
    # Create the single layer segmentation head for ImageNet-S
    seg_head = SingleLayerSegmentationHead(
        input_shape=input_shape,
        hidden_dim=256,
        num_classes=919,  # ImageNet-S has 919 semantic classes
    )
    
    # Example forward pass with 224x224 input resolution
    batch_size = 2
    features = {
        "res2": torch.randn(batch_size, 256, 56, 56),
        "res3": torch.randn(batch_size, 512, 28, 28),
        "res4": torch.randn(batch_size, 1024, 14, 14),
        "res5": torch.randn(batch_size, 2048, 7, 7),
    }
    
    output = seg_head(features)
    print("Output keys:", output.keys())
    print("Pred logits shape:", output["pred_logits"].shape)  # [2, 100, 920] (919 classes + 1 background)
    print("Pred masks shape:", output["pred_masks"].shape)   # [2, 100, 56, 56] (will be upsampled to 224x224)
    
    # Example prediction with upsampling to full resolution
    output_full_res = seg_head.predict(features, rescale_to=(224, 224))
    print("Full resolution pred masks shape:", output_full_res["pred_masks"].shape)  # [2, 100, 224, 224]