import torch.nn as nn

from .activation import Activation

# Could be replaced by torchvision implementation


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        act=None,
    ):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(
            out_channels,
        )
        self.act = act
        if self.act is not None:
            self.act = Activation(act_type=act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.relu1 = Activation(act_type="relu", inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.hard_sigmoid = Activation(act_type="hard_sigmoid", inplace=True)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = self.relu1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.hard_sigmoid(outputs)
        outputs = inputs * outputs
        return outputs


class ResidualUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        kernel_size,
        stride,
        use_se,
        act=None,
    ):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=act,
        )
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=mid_channels,
            act=act,
        )
        if self.if_se:
            self.mid_se = SEModule(mid_channels)
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            act=None,
        )

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = inputs + x
        return x


class MobileNetV3Det(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_name="large",
        scale=0.5,
        disable_se=False,
    ):
        """
        the MobilenetV3 backbone network for detection module.
        Args:
            params(dict): the super parameters for build network
        """
        super().__init__()

        self.disable_se = disable_se

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, "relu", 1],
                [3, 64, 24, False, "relu", 2],
                [3, 72, 24, False, "relu", 1],
                [5, 72, 40, True, "relu", 2],
                [5, 120, 40, True, "relu", 1],
                [5, 120, 40, True, "relu", 1],
                [3, 240, 80, False, "hard_swish", 2],
                [3, 200, 80, False, "hard_swish", 1],
                [3, 184, 80, False, "hard_swish", 1],
                [3, 184, 80, False, "hard_swish", 1],
                [3, 480, 112, True, "hard_swish", 1],
                [3, 672, 112, True, "hard_swish", 1],
                [5, 672, 160, True, "hard_swish", 2],
                [5, 960, 160, True, "hard_swish", 1],
                [5, 960, 160, True, "hard_swish", 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, "relu", 2],
                [3, 72, 24, False, "relu", 2],
                [3, 88, 24, False, "relu", 1],
                [5, 96, 40, True, "hard_swish", 2],
                [5, 240, 40, True, "hard_swish", 1],
                [5, 240, 40, True, "hard_swish", 1],
                [5, 120, 48, True, "hard_swish", 1],
                [5, 144, 48, True, "hard_swish", 1],
                [5, 288, 96, True, "hard_swish", 2],
                [5, 576, 96, True, "hard_swish", 1],
                [5, 576, 96, True, "hard_swish", 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError(
                "mode[" + model_name + "_model] is not implemented!"
            )

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, (
            "supported scale are {} but input scale is {}".format(
                supported_scale, scale
            )
        )
        inplanes = 16
        # conv1
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            act="hard_swish",
        )

        self.stages = nn.ModuleList()
        self.out_channels = []
        block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        for k, exp, c, se, nl, s in cfg:
            se = se and not self.disable_se
            start_idx = 2 if model_name == "large" else 0
            if s == 2 and i > start_idx:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                )
            )
            inplanes = make_divisible(scale * c)
            i += 1
        block_list.append(
            ConvBNLayer(
                in_channels=inplanes,
                out_channels=make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                act="hard_swish",
            )
        )
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))

    def forward(self, x):
        x = self.conv(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list


class DepthwiseSeparable(nn.Module):
    def __init__(
        self,
        in_channels,
        num_filters1,
        num_filters2,
        groups,
        stride,
        scale,
        dw_size=3,
        padding=1,
        use_se=False,
    ):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self._depthwise_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=int(num_filters1 * scale),
            kernel_size=dw_size,
            stride=stride,
            padding=padding,
            groups=int(groups * scale),
            act="hard_swish",
        )
        if use_se:
            self._se = SEModule(int(num_filters1 * scale))
        self._pointwise_conv = ConvBNLayer(
            in_channels=int(num_filters1 * scale),
            kernel_size=1,
            out_channels=int(num_filters2 * scale),
            stride=1,
            padding=0,
            act="hard_swish",
        )

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        if self.use_se:
            y = self._se(y)
        y = self._pointwise_conv(y)
        return y


class MobileNetV1Enhance(nn.Module):
    def __init__(
        self,
        in_channels=3,
        scale=0.5,
        last_conv_stride=1,
        last_pool_type="max",
    ):
        super().__init__()
        self.scale = scale
        self.block_list = []

        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            kernel_size=3,
            out_channels=int(32 * scale),
            stride=2,
            padding=1,
            act="hard_swish",
        )

        conv2_1 = DepthwiseSeparable(
            in_channels=int(32 * scale),
            num_filters1=32,
            num_filters2=64,
            groups=32,
            stride=1,
            scale=scale,
        )
        self.block_list.append(conv2_1)

        conv2_2 = DepthwiseSeparable(
            in_channels=int(64 * scale),
            num_filters1=64,
            num_filters2=128,
            groups=64,
            stride=1,
            scale=scale,
        )
        self.block_list.append(conv2_2)

        conv3_1 = DepthwiseSeparable(
            in_channels=int(128 * scale),
            num_filters1=128,
            num_filters2=128,
            groups=128,
            stride=1,
            scale=scale,
        )
        self.block_list.append(conv3_1)

        conv3_2 = DepthwiseSeparable(
            in_channels=int(128 * scale),
            num_filters1=128,
            num_filters2=256,
            groups=128,
            stride=(2, 1),
            scale=scale,
        )
        self.block_list.append(conv3_2)

        conv4_1 = DepthwiseSeparable(
            in_channels=int(256 * scale),
            num_filters1=256,
            num_filters2=256,
            groups=256,
            stride=1,
            scale=scale,
        )
        self.block_list.append(conv4_1)

        conv4_2 = DepthwiseSeparable(
            in_channels=int(256 * scale),
            num_filters1=256,
            num_filters2=512,
            groups=256,
            stride=(2, 1),
            scale=scale,
        )
        self.block_list.append(conv4_2)

        for _ in range(5):
            conv5 = DepthwiseSeparable(
                in_channels=int(512 * scale),
                num_filters1=512,
                num_filters2=512,
                groups=512,
                stride=1,
                dw_size=5,
                padding=2,
                scale=scale,
                use_se=False,
            )
            self.block_list.append(conv5)

        conv5_6 = DepthwiseSeparable(
            in_channels=int(512 * scale),
            num_filters1=512,
            num_filters2=1024,
            groups=512,
            stride=(2, 1),
            dw_size=5,
            padding=2,
            scale=scale,
            use_se=True,
        )
        self.block_list.append(conv5_6)

        conv6 = DepthwiseSeparable(
            in_channels=int(1024 * scale),
            num_filters1=1024,
            num_filters2=1024,
            groups=1024,
            stride=last_conv_stride,
            dw_size=5,
            padding=2,
            use_se=True,
            scale=scale,
        )
        self.block_list.append(conv6)

        self.block_list = nn.Sequential(*self.block_list)
        if last_pool_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * scale)

    def forward(self, inputs):
        y = self.conv1(inputs)
        y = self.block_list(y)
        y = self.pool(y)
        return y


class MobileNetV3Rec(nn.Module):
    def __init__(
        self,
        in_channels=3,
        model_name="small",
        scale=0.5,
        large_stride=None,
        small_stride=None,
    ):
        super().__init__()
        if small_stride is None:
            small_stride = [2, 2, 2, 2]
        if large_stride is None:
            large_stride = [1, 2, 2, 2]

        assert isinstance(large_stride, list), (
            "large_stride type must be list but got {}".format(type(large_stride))
        )
        assert isinstance(small_stride, list), (
            "small_stride type must be list but got {}".format(type(small_stride))
        )
        assert len(large_stride) == 4, (
            "large_stride length must be 4 but got {}".format(len(large_stride))
        )
        assert len(small_stride) == 4, (
            "small_stride length must be 4 but got {}".format(len(small_stride))
        )

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, "relu", large_stride[0]],
                [3, 64, 24, False, "relu", (large_stride[1], 1)],
                [3, 72, 24, False, "relu", 1],
                [5, 72, 40, True, "relu", (large_stride[2], 1)],
                [5, 120, 40, True, "relu", 1],
                [5, 120, 40, True, "relu", 1],
                [3, 240, 80, False, "hard_swish", 1],
                [3, 200, 80, False, "hard_swish", 1],
                [3, 184, 80, False, "hard_swish", 1],
                [3, 184, 80, False, "hard_swish", 1],
                [3, 480, 112, True, "hard_swish", 1],
                [3, 672, 112, True, "hard_swish", 1],
                [5, 672, 160, True, "hard_swish", (large_stride[3], 1)],
                [5, 960, 160, True, "hard_swish", 1],
                [5, 960, 160, True, "hard_swish", 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, "relu", (small_stride[0], 1)],
                [3, 72, 24, False, "relu", (small_stride[1], 1)],
                [3, 88, 24, False, "relu", 1],
                [5, 96, 40, True, "hard_swish", (small_stride[2], 1)],
                [5, 240, 40, True, "hard_swish", 1],
                [5, 240, 40, True, "hard_swish", 1],
                [5, 120, 48, True, "hard_swish", 1],
                [5, 144, 48, True, "hard_swish", 1],
                [5, 288, 96, True, "hard_swish", (small_stride[3], 1)],
                [5, 576, 96, True, "hard_swish", 1],
                [5, 576, 96, True, "hard_swish", 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError(
                "mode[" + model_name + "_model] is not implemented!"
            )

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, (
            "supported scales are {} but input scale is {}".format(
                supported_scale, scale
            )
        )

        inplanes = 16
        # conv1
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            act="hard_swish",
        )
        i = 0
        block_list = []
        inplanes = make_divisible(inplanes * scale)
        for k, exp, c, se, nl, s in cfg:
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                )
            )
            inplanes = make_divisible(scale * c)
            i += 1
        self.blocks = nn.Sequential(*block_list)

        self.conv2 = ConvBNLayer(
            in_channels=inplanes,
            out_channels=make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            act="hard_swish",
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = make_divisible(scale * cls_ch_squeeze)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x
