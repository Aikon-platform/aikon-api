import torch
from torch import nn
import math
import numpy as np

import torch.nn.functional as F
import torchvision.models as models

from PIL import Image

from .const import MAX_SIZE, SEG_STRIDE, SEG_TOPK

"""
MIT License

Copyright (c) 2021 xshen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Mostly copy-paste from https://github.com/XiSHEN0220/SegSwap

@article{shen2021learning,
  title={Learning Co-segmentation by Segment Swapping for Retrieval and Discovery},
  author={Shen, Xi and Efros, Alexei A and Joulin, Armand and Aubry, Mathieu},
  journal={arXiv},
  year={2021}
}
"""

# ── Shared forward pass for encoder layers ───────────────────────────────────


def frwd(featx, featy, encoder_layer, self_attn_only=True, x_mask=None, y_mask=None):
    """
    Shared forward logic for inner-attention and cross-attention encoder layers.
    Device is inferred from input tensors — no hardcoded CUDA dependency.

    self_attn_only=True  → inner attention (block-diagonal mask, each image attends only to itself)
    self_attn_only=False → cross attention (off-diagonal mask, each image attends only to the other)
    """
    bx, cx, hx, wx = featx.size()
    by, cy, hy, wy = featy.size()
    device = featx.device

    ## input of transformer should be : seq_len * batch_size * feat_dim
    featx = featx.flatten(2).permute(2, 0, 1)
    featy = featy.flatten(2).permute(2, 0, 1)

    x_mask = (
        x_mask.flatten(2).squeeze(1)
        if x_mask is not None
        else torch.zeros(bx, hx * wx, dtype=torch.bool, device=device)
    )
    y_mask = (
        y_mask.flatten(2).squeeze(1)
        if y_mask is not None
        else torch.zeros(by, hy * wy, dtype=torch.bool, device=device)
    )

    ## input of transformer: (seq_len*2) * batch_size * feat_dim
    len_seq_x, len_seq_y = featx.size()[0], featy.size()[0]

    output = torch.cat([featx, featy], dim=0)
    src_key_padding_mask = torch.cat((x_mask, y_mask), dim=1)

    total = hx * wx + hy * wy
    with torch.no_grad():
        src_mask = torch.full(
            (total, total), self_attn_only, dtype=torch.bool, device=device
        )
        src_mask[: hx * wx, : hx * wx] = not self_attn_only
        src_mask[hx * wx :, hx * wx :] = not self_attn_only

    output = encoder_layer(
        output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
    )

    outx = output.narrow(0, 0, len_seq_x).permute(1, 2, 0).view(bx, cx, hx, wx)
    outy = output.narrow(0, len_seq_x, len_seq_y).permute(1, 2, 0).view(by, cy, hy, wy)
    x_mask = x_mask.view(bx, 1, hx, wx)
    y_mask = y_mask.view(by, 1, hy, wy)
    return outx, outy, x_mask, y_mask


# --- Positional encoding--- #
# --- Borrowed from Detr--- #


def consistent_mask(m1, m2, o1, o2):
    """
    Produce masks that are cycle-consistent with flow

    Input: masks (m1 and m2) and flows (o1 and o2) predicted for both images
    Output: masks that are cycle-consistent with flow prediction (m1_final, m2_final)
    """
    flow12 = (o1.narrow(1, 0, 2).permute(0, 2, 3, 1) - 0.5) * 2
    flow21 = (o2.narrow(1, 0, 2).permute(0, 2, 3, 1) - 0.5) * 2

    m1_back = F.grid_sample(
        torch.from_numpy(m2.astype(np.float32)),
        flow12,
        mode="bilinear",
        align_corners=False,
    )
    m1_final = m1 * m1_back.numpy()

    m2_back = F.grid_sample(
        torch.from_numpy(m1.astype(np.float32)),
        flow21,
        mode="bilinear",
        align_corners=False,
    )
    m2_final = m2 * m2_back.numpy()
    return m1_final, m2_final


def resize(img: Image, img_size=MAX_SIZE, stride=SEG_STRIDE):
    """
    Resize images
    1. Each dimension can be divided by stride (16)
    2. largest dimension is defined by img_size
    3. keeping aspect ratio

    Input: masks (m1 and m2) and flows (o1 and o2) predicted for both images
    Output: masks that are cycle-consistent with flow prediction (m1_final, m2_final)
    """
    w, h = img.size
    ratio = max(1.0 * w / img_size, 1.0 * h / img_size)

    new_w = int(round(w / ratio / stride)) * stride
    new_h = int(round(h / ratio / stride)) * stride

    return img.resize((new_w, new_h), resample=2)


def score_local_feat_match(feat1, x1, y1, feat2, x2, y2, weight_feat):
    with torch.no_grad():  # NOTE it it necessary since the function is called inside a with torch.no_grad() block
        feat_2_bag = []
        feat_1_bag = []
        for j in range(feat1.shape[0]):
            feat_2_bag_ = [
                feat2[j, :, y2[j, i], x2[j, i]].unsqueeze(0) for i in range(x2.shape[1])
            ]
            feat_2_bag.append(torch.cat(feat_2_bag_, dim=0))

            feat_1_bag_ = [
                feat1[j, :, y1[i], x1[i]].unsqueeze(0) for i in range(x1.shape[0])
            ]
            feat_1_bag.append(torch.cat(feat_1_bag_, dim=0))

        feat_1_bag = torch.stack(feat_1_bag)
        feat_2_bag = torch.stack(feat_2_bag)
        # each local feature is matched to its feature in another image,
        # finally the similarity is weighted by the mask prediction
        score_weight_feat = torch.sum(feat_1_bag * feat_2_bag, dim=2) * weight_feat

        # if (x1i, y1i) --> (x2, y2)
        # and (x1j, y1j) --> (x2, y2)
        # pick the best match
        scores = []
        for j in range(feat1.shape[0]):
            dict_score = {
                (y2[j, i].item(), x2[j, i].item()): [0] for i in range(x2.shape[1])
            }
            for i in range(x2.shape[1]):
                dict_score[(y2[j, i].item(), x2[j, i].item())].append(
                    score_weight_feat[j, i].item()
                )
            scores.append(sum([np.max(dict_score[key]) for key in dict_score]))
        return np.array(scores)


def load_backbone(param):
    backbone = models.resnet50(weights=None)
    resnet_feature_layers = [
        "conv1",
        "bn1",
        "relu",
        "maxpool",
        "layer1",
        "layer2",
        "layer3",
    ]
    resnet_module_list = [getattr(backbone, l) for l in resnet_feature_layers]
    last_layer_idx = resnet_feature_layers.index("layer3")
    backbone = torch.nn.Sequential(*resnet_module_list[: last_layer_idx + 1])
    backbone.load_state_dict(param["backbone"])
    backbone.eval()
    return backbone


def load_encoder(param):
    net_encoder = TransEncoder(
        feat_dim=1024,
        pos_weight=0.1,
        feat_weight=1,
        dropout=0.1,
        activation="relu",
        mode="small",
        layer_type=["I", "C", "I", "C", "I", "N"],
        drop_feat=0.1,
    )
    net_encoder.load_state_dict(param["encoder"])
    net_encoder.eval()
    return net_encoder


@torch.no_grad()
def compute_score(
    tensor1,
    tensor2,
    backbone,
    net_encoder,
    y_grid,
    x_grid,
    nb_feat_h=MAX_SIZE // SEG_STRIDE,
    nb_feat_w=MAX_SIZE // SEG_STRIDE,
):
    feat1 = F.normalize(backbone(tensor1))
    feat2 = F.normalize(backbone(tensor2))
    out1, out2 = net_encoder(feat1, feat2)

    m1_final, m2_final = consistent_mask(
        out1[:, 2:].cpu().numpy(),
        out2[:, 2:].cpu().numpy(),
        out1.cpu(),
        out2.cpu(),
    )
    x2_pred = np.round(
        out1[:, 0, y_grid, x_grid].cpu().numpy() * (nb_feat_w - 1)
    ).astype(int)
    y2_pred = np.round(
        out1[:, 1, y_grid, x_grid].cpu().numpy() * (nb_feat_h - 1)
    ).astype(int)

    m1_final_ = m1_final[:, 0, y_grid, x_grid]
    return score_local_feat_match(
        feat1.cpu(), x_grid, y_grid, feat2.cpu(), x2_pred, y2_pred, m1_final_
    )


def save_mask(q_img, sim_imgs, img_dir, m2_final):
    # NOT USED
    arr2 = np.array([np.array(I2) for I2 in sim_imgs])
    m2_up = F.interpolate(
        torch.from_numpy(m2_final.astype(np.float32)),
        size=(arr2.shape[1], arr2.shape[2]),
        mode="bilinear",
    ).numpy()
    m2_up = np.clip(m2_up * 255 + 50, a_min=50, a_max=255)
    m2_up = m2_up.astype(np.uint8).reshape(SEG_TOPK, arr2.shape[1], arr2.shape[2], 1)
    arr2_mask = np.concatenate((arr2, m2_up), axis=3)

    new_images = [Image.fromarray(img) for img in arr2_mask]
    for i in range(len(new_images)):
        new_images[i].save(f"{img_dir}/{q_img.split('.')[0]}_masked_{i}.png")


class PositionEncodingSine2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super(PositionEncodingSine2D, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, isTarget=False):
        """
        input x: B, C, H, W
        return pos: B, C, H, W
        """
        not_mask = torch.ones(x.size()[0], x.size()[2], x.size()[3]).to(x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            ## no diff between source and target

            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class EncoderLayerInnerAttention(nn.Module):
    """
    Transformer encoder with all parameters
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        pos_weight,
        feat_weight,
    ):
        super(EncoderLayerInnerAttention, self).__init__()

        self.pos_weight = pos_weight
        self.feat_weight = feat_weight
        self.inner_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.posEncoder = PositionEncodingSine2D(d_model // 2)

    def forward(self, x, y, x_mask=None, y_mask=None):
        """
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        """

        posx = self.posEncoder(x)
        posy = self.posEncoder(y)

        featx = self.feat_weight * x + self.pos_weight * posx
        featy = self.feat_weight * y + self.pos_weight * posy

        return frwd(
            featx,
            featy,
            self.inner_encoder_layer,
            self_attn_only=True,
            x_mask=x_mask,
            y_mask=y_mask,
        )


class EncoderLayerCrossAttention(nn.Module):
    """
    Transformer encoder with all paramters
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(EncoderLayerCrossAttention, self).__init__()

        self.cross_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, featx, featy, x_mask=None, y_mask=None):
        """
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        """

        return frwd(
            featx,
            featy,
            self.cross_encoder_layer,
            self_attn_only=False,
            x_mask=x_mask,
            y_mask=y_mask,
        )


class EncoderLayerEmpty(nn.Module):
    """
    Transformer encoder with all parameters
    """

    def __init__(self):
        super(EncoderLayerEmpty, self).__init__()

    def forward(self, featx, featy, x_mask=None, y_mask=None):
        """
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        """

        return featx, featy, x_mask, y_mask


class EncoderLayerBlock(nn.Module):
    """
    Transformer encoder with all parameters
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        activation,
        pos_weight,
        feat_weight,
        layer_type,
    ):
        super(EncoderLayerBlock, self).__init__()

        cross_encoder_layer = EncoderLayerCrossAttention(
            d_model, nhead, dim_feedforward, dropout, activation
        )
        att_encoder_layer = EncoderLayerInnerAttention(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            pos_weight,
            feat_weight,
        )

        if layer_type[0] == "C":
            self.layer1 = cross_encoder_layer
        elif layer_type[0] == "I":
            self.layer1 = att_encoder_layer
        elif layer_type[0] == "N":
            self.layer1 = EncoderLayerEmpty()

        if layer_type[1] == "C":
            self.layer2 = cross_encoder_layer
        elif layer_type[1] == "I":
            self.layer2 = att_encoder_layer
        elif layer_type[1] == "N":
            self.layer2 = EncoderLayerEmpty()

    def forward(self, featx, featy, x_mask=None, y_mask=None):
        """
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        """

        featx, featy, x_mask, y_mask = self.layer1(featx, featy, x_mask, y_mask)
        featx, featy, x_mask, y_mask = self.layer2(featx, featy, x_mask, y_mask)

        return featx, featy, x_mask, y_mask


### --- Transformer Encoder --- ###


class Encoder(nn.Module):
    """
    Transformer encoder with all parameters
    """

    def __init__(
        self,
        feat_dim,
        pos_weight=0.1,
        feat_weight=1,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_type=["I", "C", "I", "C", "I", "C"],
        drop_feat=0.1,
    ):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.feat_proj = nn.Conv2d(feat_dim, d_model, kernel_size=1)
        self.drop_feat = nn.Dropout2d(p=drop_feat)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderLayerBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout,
                    activation,
                    pos_weight,
                    feat_weight,
                    layer_type[i * 2 : i * 2 + 2],
                )
                for i in range(num_layers)
            ]
        )

        self.final_linear = nn.Conv2d(d_model, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.eps = 1e-7

    def forward(self, x, y, x_mask=None, y_mask=None):
        """
        input x: B, C, H, W
        input y: B, C, H, W
        input x_mask: B, 1, H, W, mask == True will be ignored
        input y_mask: B, 1, H, W, mask == True will be ignored
        """
        featx = self.feat_proj(x)
        featx = self.drop_feat(featx)

        # bx, cx, hx, wx = featx.size()

        featy = self.feat_proj(y)
        featy = self.drop_feat(featy)

        # by, cy, hy, wy = featy.size()
        for i in range(self.num_layers):
            featx, featy, x_mask, y_mask = self.encoder_blocks[i](
                featx, featy, x_mask, y_mask
            )

        outx = self.sigmoid(self.final_linear(featx))
        outy = self.sigmoid(self.final_linear(featy))

        outx = torch.clamp(outx, min=self.eps, max=1 - self.eps)
        outy = torch.clamp(outy, min=self.eps, max=1 - self.eps)

        return outx, outy


### --- Transformer Encoder --- ###


class TransEncoder(nn.Module):
    """
    Transformer encoder: small and large variants
    """

    def __init__(
        self,
        feat_dim=1024,
        pos_weight=0.1,
        feat_weight=1,
        dropout=0.1,
        activation="relu",
        mode="small",
        layer_type=["I", "C", "I", "C", "I", "N"],
        drop_feat=0.1,
    ):
        super(TransEncoder, self).__init__()

        if mode == "tiny":
            d_model = 128
            nhead = 2
            num_layers = 3
            dim_feedforward = 256

        elif mode == "small":
            d_model = 256
            nhead = 2
            num_layers = 3
            dim_feedforward = 256

        elif mode == "base":
            d_model = 512
            nhead = 8
            num_layers = 3
            dim_feedforward = 2048

        elif mode == "large":
            d_model = 512
            nhead = 8
            num_layers = 6
            dim_feedforward = 2048

        self.net = Encoder(
            feat_dim,
            pos_weight,
            feat_weight,
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
            activation,
            layer_type,
            drop_feat,
        )

    def forward(self, x, y, x_mask=None, y_mask=None):
        """
        input x: B, C, H, W
        input y: B, C, H, W
        """
        outx, outy = self.net(x, y, x_mask, y_mask)
        return outx, outy
