import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_feature(x, size=None, scale_factor=None, mode="bilinear", align_corners=False):
    return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


def _make_norm(num_channels, norm="gn"):
    if norm == "bn":
        return nn.BatchNorm2d(num_channels)
    if norm != "gn":
        raise ValueError(f"Unsupported norm '{norm}'.")
    num_groups = min(32, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=max(1, num_groups), num_channels=num_channels)


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        norm="gn",
        activation=True,
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            _make_norm(out_channels, norm=norm),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class StandardMultiLevelNeck(nn.Module):
    def __init__(self, in_channels, out_channels, scales=None, norm="gn"):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.scales = list(scales or [1.0] * len(self.in_channels))
        self.lateral_convs = nn.ModuleList(
            [ConvNormAct(channels, out_channels, kernel_size=1, norm=norm) for channels in self.in_channels]
        )
        self.output_convs = nn.ModuleList(
            [ConvNormAct(out_channels, out_channels, kernel_size=3, padding=1, norm=norm) for _ in self.scales]
        )

    def forward(self, inputs):
        if len(inputs) != len(self.in_channels):
            raise ValueError(f"Expected {len(self.in_channels)} feature levels, got {len(inputs)}.")
        features = [conv(feat) for conv, feat in zip(self.lateral_convs, inputs)]
        if len(features) == 1:
            features = [features[0] for _ in self.scales]
        outputs = []
        for idx, scale in enumerate(self.scales):
            feat = features[min(idx, len(features) - 1)]
            if scale != 1.0:
                target_h = max(1, int(round(feat.shape[-2] * scale)))
                target_w = max(1, int(round(feat.shape[-1] * scale)))
                feat = resize_feature(feat, size=(target_h, target_w))
            outputs.append(self.output_convs[idx](feat))
        return outputs


class PyramidPoolingModule(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, channels, norm="gn", align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvNormAct(in_channels, channels, kernel_size=1, norm=norm),
                )
            )

    def forward(self, x):
        outputs = []
        for stage in self:
            pooled = stage(x)
            outputs.append(
                resize_feature(pooled, size=x.shape[-2:], mode="bilinear", align_corners=self.align_corners)
            )
        return outputs


class UPerNetHead(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        num_classes,
        pool_scales=(1, 2, 3, 6),
        dropout=0.1,
        norm="gn",
        align_corners=False,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = channels
        self.align_corners = align_corners
        self.psp_modules = PyramidPoolingModule(
            pool_scales,
            in_channels=self.in_channels[-1],
            channels=channels,
            norm=norm,
            align_corners=align_corners,
        )
        self.psp_bottleneck = ConvNormAct(
            self.in_channels[-1] + len(pool_scales) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm=norm,
        )
        self.lateral_convs = nn.ModuleList(
            [ConvNormAct(in_ch, channels, kernel_size=1, norm=norm) for in_ch in self.in_channels[:-1]]
        )
        self.fpn_convs = nn.ModuleList(
            [ConvNormAct(channels, channels, kernel_size=3, padding=1, norm=norm) for _ in self.in_channels[:-1]]
        )
        self.fpn_bottleneck = ConvNormAct(
            len(self.in_channels) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm=norm,
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def psp_forward(self, x):
        outputs = [x]
        outputs.extend(self.psp_modules(x))
        return self.psp_bottleneck(torch.cat(outputs, dim=1))

    def forward_features(self, inputs):
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, inputs[:-1])]
        laterals.append(self.psp_forward(inputs[-1]))
        for idx in range(len(laterals) - 1, 0, -1):
            laterals[idx - 1] = laterals[idx - 1] + resize_feature(
                laterals[idx],
                size=laterals[idx - 1].shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        fpn_outs = [conv(feat) for conv, feat in zip(self.fpn_convs, laterals[:-1])]
        fpn_outs.append(laterals[-1])
        for idx in range(1, len(fpn_outs)):
            fpn_outs[idx] = resize_feature(
                fpn_outs[idx],
                size=fpn_outs[0].shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        return self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))

    def forward(self, inputs):
        features = self.forward_features(inputs)
        return self.classifier(self.dropout(features))


class FCNHead(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        num_classes,
        in_index=2,
        num_convs=1,
        kernel_size=3,
        concat_input=False,
        dilation=1,
        dropout=0.1,
        norm="gn",
    ):
        super().__init__()
        self.in_index = in_index
        conv_padding = (kernel_size // 2) * dilation
        layers = []
        current_channels = in_channels
        for _ in range(max(1, num_convs)):
            layers.append(
                ConvNormAct(
                    current_channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    norm=norm,
                )
            )
            current_channels = channels
        self.convs = nn.Sequential(*layers) if num_convs > 0 else nn.Identity()
        self.concat_input = concat_input
        if concat_input:
            self.conv_cat = ConvNormAct(in_channels + channels, channels, kernel_size=kernel_size, padding=kernel_size // 2, norm=norm)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward_features(self, inputs):
        x = inputs[self.in_index]
        features = self.convs(x)
        if self.concat_input:
            features = self.conv_cat(torch.cat([x, features], dim=1))
        return features

    def forward(self, inputs):
        features = self.forward_features(inputs)
        return self.classifier(self.dropout(features))


class DinoMMUPerNetSegmentor(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes,
        feature_dim,
        neck_scales=None,
        aux_index=2,
        dropout=0.1,
        norm="gn",
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = StandardMultiLevelNeck(
            in_channels=[backbone.out_channels] * 4,
            out_channels=feature_dim,
            scales=neck_scales or [1.0, 1.0, 1.0, 1.0],
            norm=norm,
        )
        self.decode_head = UPerNetHead(
            in_channels=[feature_dim] * 4,
            channels=feature_dim,
            num_classes=num_classes,
            dropout=dropout,
            norm=norm,
        )
        self.auxiliary_head = FCNHead(
            in_channels=feature_dim,
            channels=feature_dim,
            num_classes=num_classes,
            in_index=aux_index,
            num_convs=1,
            concat_input=False,
            dropout=dropout,
            norm=norm,
        )

    def forward_features(self, x):
        features = self.backbone.forward_feature_list(x)
        return self.neck(features)

    def forward_with_aux(self, x):
        features = self.forward_features(x)
        logits = self.decode_head(features)
        aux_logits = self.auxiliary_head(features)
        logits = resize_feature(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        aux_logits = resize_feature(aux_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits, aux_logits

    def forward(self, x):
        logits, _ = self.forward_with_aux(x)
        return logits
