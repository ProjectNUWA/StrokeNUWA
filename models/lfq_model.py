import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from models.bottleneck import NoBottleneck, Bottleneck
from models.encdec import Encoder, Decoder
from vector_quantize_pytorch import ResidualLFQ

# helper functions
def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"


def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]


def average_metrics(_metrics):
    metrics = {}
    for _metric in _metrics:
        for key, val in _metric.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(val)
    return {key: sum(vals)/len(vals) for key, vals in metrics.items()}


def remove_padding(x, padding_mask):
    """
    x: (batch_size x) seq_len num_bins
    padding: seq_len x num_bins

    if batch, return List[Tensor]
    if tensor, return Tensor
    """
    if x.ndim == 2:  # seq_len x num_bins
        return x[:padding_mask.sum(), :]
    elif x.ndim == 3:  # batch_size x seq_len x num_bins
        res = []
        for i in range(x.size(0)):
            res.append(x[i, :padding_mask[i].sum(), :])
        return res


def postprocess(x, padding_mask=None, path_interpolation=True):
    """
    postprocess the generated results

    x: batch_size x seq_len x 9
    padding_mask: batch_size x seq_len
    path_interpolation: whether to interpolate the path
    """
    dtype = x.dtype
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if padding_mask is not None and padding_mask.ndim == 1:
        padding_mask = padding_mask.unsqueeze(0)

    if path_interpolation:  
        # conduct path interpolation
        # return List[Tensor]
        batch_size = x.size(0)
        if padding_mask is not None:
            x = remove_padding(x, padding_mask)  # remove the padding
        full_x = []
        for i in range(batch_size):
            current_path = []
            last_x3, last_y3 = None, None
            seq_len = x[i].size(0)
            for j in range(seq_len):
                row = x[i][j]
                cmd = 100 * t.round(row[0] / 100).item()
                cmd = 1 if cmd == 100 else 2 if cmd == 200 else 0
                x0, y0, x1, y1, x2, y2, x3, y3 = map(lambda coord: min(max(coord, 0), 200), row[1:].tolist())
                if last_x3 is not None and (last_x3 != x0 or last_y3 != y0):
                    # if the current row's start point is not the same as the previous row's end point
                    current_path.append([0, last_x3, last_y3, 0, 0, 0, 0, x0, y0])
                if cmd in [0, 100]:
                    # if the current row is M or L, set control point to 0
                    x1, y1, x2, y2 = 0, 0, 0, 0
                current_path.append([cmd, x0, y0, x1, y1, x2, y2, x3, y3])
                last_x3, last_y3 = x3, y3  # update the last end point
            full_x.append(t.tensor(current_path, dtype=dtype))
    
    else:  # no path interpolation
        if x.size(-1) == 9:
            # first remove the 1, 2 columns
            m_x = t.cat((x[:, :, :1], x[:, :, 3:]), dim=2)
        else:
            m_x = x
        # find the right command value
        m_x[:, :, 0] = t.round(m_x[:, :, 0] / 100) * 100
        # clip all the value to max bins 
        m_x = t.clamp(m_x, 0, 200)
        # process the M and L path                                                              
        m_x[:, :, 1:5][m_x[:, :, 0] != 200] = 0
        # add to extra column to satisfy the 9 columns
        x_0_y_0 = t.zeros((m_x.size(0), m_x.size(1), 2), dtype=m_x.dtype, device=m_x.device)
        x_0_y_0[:, 1:, 0] = m_x[:, :-1, -2]  # x_3 of the previous row
        x_0_y_0[:, 1:, 1] = m_x[:, :-1, -1]  # y_3 of the previous row
        full_x = t.cat((m_x[:, :, :1], x_0_y_0, m_x[:, :, 1:]), dim=2)
        # replace the command value to 0, 1, 2
        full_x[:, :, 0][full_x[:, :, 0] == 100] = 1
        full_x[:, :, 0][full_x[:, :, 0] == 200] = 2
        # remove the padding
        full_x = remove_padding(full_x, padding_mask)

    return full_x


def _loss_fn(loss_fn, x_target, x_pred, cfg, padding_mask=None):
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(-1).expand_as(x_target)
        x_target = t.where(padding_mask, x_target, t.zeros_like(x_target)).to(x_pred.device)
        x_pred = t.where(padding_mask, x_pred, t.zeros_like(x_pred)).to(x_pred.device)
        mask_sum = padding_mask.sum()

    if loss_fn == 'l1':
        loss = t.sum(t.abs(x_pred - x_target)) / mask_sum
    elif loss_fn == 'l2':
        loss = t.sum((x_pred - x_target) ** 2) / mask_sum
    elif loss_fn == 'linf':
        residual = ((x_pred - x_target) ** 2).reshape(x_target.shape[0], -1)
        # only consider the residual of the padded part
        masked_residual = t.where(padding_mask.reshape(x_target.shape[0], -1), residual, t.zeros_like(residual))
        values, _ = t.topk(masked_residual, cfg.linf_k, dim=1)
        loss = t.mean(values)
    else:
        assert False, f"Unknown loss_fn {loss_fn}"

    return loss


class LFQ(nn.Module):
    def __init__(self, config, multipliers=None, **block_kwargs):
        super().__init__()
        self.cfg = config.lfq
        self.vocab_size = config.dataset.vocab_size
        self.commit = self.cfg.commit
        self.recon = self.cfg.recon
        self.sample_length = config.dataset.max_path_nums
        self.x_channels = config.dataset.x_channels
        self.x_shape = (config.dataset.max_path_nums, config.dataset.x_channels)
        self.num_quantizers = self.cfg.num_quantizers
        self.levels = self.cfg.levels
        self.codebook_size = self.cfg.codebook_size
        if multipliers is None:
            self.multipliers = [1] * self.levels
        else:
            assert len(multipliers) == self.num_quantizers, "Invalid number of multipliers"
            self.multipliers = multipliers

        # define encoder and decoder
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs
        
        def encoder(level): 
            return Encoder(
                input_emb_width=self.x_channels,
                output_emb_width=self.cfg.emb_width,
                levels=level + 1,
                downs_t=self.cfg.downs_t[:level+1],
                strides_t=self.cfg.strides_t[:level+1],
                use_modified_block=self.cfg.use_modified_block,
                **_block_kwargs(level)
            )
        
        def decoder(level): 
            return Decoder(
                input_emb_width = self.x_channels, 
                output_emb_width = self.cfg.emb_width, 
                levels = level + 1,
                downs_t = self.cfg.downs_t[:level+1], 
                strides_t = self.cfg.strides_t[:level+1], 
                use_modified_block=self.cfg.use_modified_block,
                **_block_kwargs(level)
            )

        for level in range(self.cfg.levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))

        # define bottleneck
        if self.cfg.use_bottleneck:
            self.bottleneck = ResidualLFQ(
                dim = self.cfg.emb_width,
                codebook_size = self.codebook_size,
                num_quantizers = self.num_quantizers
            )
        else:
            self.bottleneck = NoBottleneck(self.cfg.levels)    

    def normalize_func(self, tensor, min_val=0, max_val=200):
        # normalize to [-1, 1]
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        normalized_tensor = normalized_tensor * 2 - 1
        return normalized_tensor


    def denormalize_func(self, normalized_tensor, min_val=0, max_val=200):
        tensor = (normalized_tensor + 1) / 2
        tensor = tensor * (max_val - min_val) + min_val
        tensor = t.round(tensor).long()
        return tensor
        

    def decode(self, zs, start_level=0, end_level=None, padding_mask=None, path_interpolation=False, return_postprocess=True):
        if end_level is None:
            end_level = self.levels
            
        xs_quantised = self.bottleneck.decode(zs, start_level=start_level, end_level=end_level)
        
        # Use only lowest level
        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False).permute(0, 2, 1)
        x_out = self.denormalize_func(x_out)

        if return_postprocess:
            x_out = postprocess(x_out, padding_mask, path_interpolation)  
        return x_out


    def encode(self, x, start_level=0, end_level=None):
        x = self.normalize_func(x) # normalize to [-1, 1]
        x_in = x.permute(0, 2, 1)  # x_in (32, 9, 256)
        xs = []

        if end_level is None:
            end_level = self.levels

        for level in range(self.levels):
            x_out = self.encoders[level](x_in)
            xs.append(x_out[-1])
            
        zs = self.bottleneck.encode(xs[start_level:end_level])
        return zs
    
    @t.no_grad()
    def encode_no_grad(self, x, start_level=0, end_level=None):  # for deepspeed hf Trainer
        x = self.normalize_func(x) # normalize to [-1, 1]
        x_in = x.permute(0, 2, 1)  # x_in (32, 9, 256)
        xs = []

        if end_level is None:
            end_level = self.levels

        for level in range(self.levels):
            x_out = self.encoders[level](x_in)
            xs.append(x_out[-1])
            
        zs = self.bottleneck.encode(xs[start_level:end_level])
        return zs


    def sample(self, n_samples):  # random sample from prior
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape),
                        device='cuda') for z_shape in self.z_shapes]
        return self.decode(zs)


    def forward(self, x, padding_mask=None, loss_fn='l2', return_all_quantized_res=False, denormalize=False):
        """
        x: [B, L, C]
        padding_mask: [B, L]
        """
        
        x = self.normalize_func(x, 0, self.vocab_size) # normalize to [-1, 1]
        # x_in = x.permute(0, 2, 1).float()  # x_in (32, 9, 256)
        x_in = x.permute(0, 2, 1)
        xs = []

        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
            # xs: [[32, 2048, 128], [32, 2048, 64], [32, 2048, 32]]

        xs_quantised, commit_losses, zs = [], [], []
        for level in range(self.levels):
            quantized, indices, commit_loss = self.bottleneck(xs[level].permute(0, 2, 1))
            xs_quantised.append(quantized.permute(0, 2, 1))
            commit_losses.append(commit_loss)
            zs.append(indices)
            # xs_quantised: [[32, 2048, 128], [32, 2048, 64], [32, 2048, 32]]

        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level+1], all_levels=False)

            # happens when deploying
            if (x_out.shape != x_in.shape):
                x_out = F.pad(
                    input=x_out, 
                    pad=(0, x_in.shape[-1]-x_out.shape[-1]), 
                    mode='constant', 
                    value=0
                )

            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out) # x_outs: [[32, 9, 256], [32, 9, 256], [32, 9, 256]]
        
        x_out, loss = None, None
        metrics = {}
        
        recons_loss = t.zeros(()).to(x.device)
        x_target = x.float()

        for level in reversed(range(self.levels)):  # attention: here utilize the reversed order
            
            x_out = x_outs[level].permute(0, 2, 1).float()
            this_recons_loss = _loss_fn(loss_fn, x_target, x_out, self.cfg, padding_mask)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            recons_loss += this_recons_loss 

        commit_loss = sum(commit_losses)
        loss = self.recon * recons_loss + self.commit * commit_loss 

        with t.no_grad():
            l2_loss = _loss_fn("l2", x_target, x_out, self.cfg, padding_mask)
            l1_loss = _loss_fn("l1", x_target, x_out, self.cfg, padding_mask)

        metrics.update(dict(
            recons_loss=recons_loss,
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            commit_loss=commit_loss,
        ))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        if return_all_quantized_res:
            x_outs = [tmp.permute(0, 2, 1) for tmp in x_outs]
            if denormalize:
                x_outs = [self.denormalize_func(tmp, 0, self.vocab_size) for tmp in x_outs]
            return x_outs, zs[0], xs_quantised[0]

        return x_out, loss, metrics

