import torch as t
import torch.nn as nn
from models.resnet import Resnet1D

# helper functions
def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"

def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]


class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, width, depth, m_conv,dilation_growth_rate=1, dilation_cycle=None,zero_out=False, res_scale=False):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(
                    nn.Conv1d(
                        in_channels= input_emb_width if i == 0 else width,
                        out_channels = width, 
                        kernel_size = filter_t, 
                        stride = stride_t, 
                        padding = pad_t,
                        dilation = dilation_growth_rate,
                    ),
                    Resnet1D(
                        n_in = width,  # 128 input embedding
                        n_depth = depth,  # 3 depth of resnet
                        m_conv = m_conv,  # 1.0  kernel size multiplier
                        dilation_growth_rate = dilation_growth_rate,
                        dilation_cycle = dilation_cycle,
                        zero_out = zero_out,
                        res_scale = res_scale
                    ),
                )
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class ModifiedEncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t, stride_t, width, depth, m_conv,dilation_growth_rate=1, dilation_cycle=None,zero_out=False, res_scale=False):
        super().__init__()
        blocks = []
        filter_t, pad_t = stride_t, 0
        if down_t > 0:
            for i in range(down_t):
                block = nn.Sequential(
                    nn.Conv1d(
                        in_channels= input_emb_width if i == 0 else width,
                        out_channels = width, 
                        kernel_size = filter_t, 
                        stride = stride_t, 
                        padding = pad_t,
                        dilation = 1,
                    ),
                    Resnet1D(
                        n_in = width,  # 128 input embedding
                        n_depth = depth,  # 3 depth of resnet
                        m_conv = m_conv,  # 1.0  kernel size multiplier
                        dilation_growth_rate = dilation_growth_rate,
                        dilation_cycle = dilation_cycle,
                        zero_out = zero_out,
                        res_scale = res_scale
                    ),
                )
                blocks.append(block)
            block = nn.Conv1d(width, output_emb_width, 2, 1, 1, 2)
            blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class ModifiedDecoderConvBock(nn.Module):
    def __init__(
        self, input_emb_width, output_emb_width, down_t, stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_decoder_dilation=False
    ):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t, 0
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(
                    Resnet1D(
                        n_in=width, 
                        n_depth=depth, 
                        m_conv=m_conv, 
                        dilation_growth_rate=dilation_growth_rate, 
                        dilation_cycle=dilation_cycle, 
                        zero_out=zero_out,
                        res_scale=res_scale, 
                        reverse_dilation=reverse_decoder_dilation, 
                    ),
                    nn.ConvTranspose1d(
                        in_channels=width,
                        out_channels=input_emb_width if i == (down_t - 1) else width,
                        kernel_size=filter_t,
                        stride=stride_t,
                        padding=pad_t,
                    ),
                )
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class DecoderConvBock(nn.Module):
    def __init__(
        self, input_emb_width, output_emb_width, down_t, stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_decoder_dilation=False
    ):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            blocks.append(block)
            for i in range(down_t):
                block = nn.Sequential(
                    Resnet1D(
                        n_in=width, 
                        n_depth=depth, 
                        m_conv=m_conv, 
                        dilation_growth_rate=dilation_growth_rate, 
                        dilation_cycle=dilation_cycle, 
                        zero_out=zero_out,
                        res_scale=res_scale, 
                        reverse_dilation=reverse_decoder_dilation, 
                    ),
                    nn.ConvTranspose1d(
                        in_channels=width,
                        out_channels=input_emb_width if i == (down_t - 1) else width,
                        kernel_size=filter_t,
                        stride=stride_t,
                        padding=pad_t,     
                    ),
                )
                blocks.append(block)
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, use_modified_block=False, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']

        def level_block(level, down_t, stride_t): 
            if use_modified_block:
                return ModifiedEncoderConvBlock(
                    input_emb_width if level == 0 else output_emb_width,
                    output_emb_width,
                    down_t, stride_t,
                    **block_kwargs_copy
                )
            else:
                return EncoderConvBlock(
                    input_emb_width if level == 0 else output_emb_width,
                    output_emb_width,
                    down_t, stride_t,
                    **block_kwargs_copy
                )
            
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []

        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            xs.append(x)

        return xs


class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t, strides_t, use_modified_block=False, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        def level_block(level, down_t, stride_t): 
            if use_modified_block:
                return ModifiedDecoderConvBock(
                    output_emb_width,
                    output_emb_width,
                    down_t, stride_t,
                    **block_kwargs
                )
            else:
                return DecoderConvBock(
                    output_emb_width,
                    output_emb_width,
                    down_t, stride_t,
                    **block_kwargs
                )
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        iterator = reversed(
            list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * (stride_t ** down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x