# -----------------------------------------------------------------------------------
# SwinIR-Strip: Image Restoration Using Swin Transformer with Strip Attention
# Based on SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Modified to include vertical strip attention for limited-angle artifact reduction
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers.drop import DropPath
from timm.layers.helpers import to_2tuple
from timm.layers.weight_init import trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def vertical_strip_partition(x, strip_width):
    """
    Partition feature map into vertical strips.
    
    Args:
        x: (B, H, W, C)
        strip_width (int): width of each strip

    Returns:
        strips: (num_strips*B, H, strip_width, C)
    """
    B, H, W, C = x.shape
    num_strips = W // strip_width
    x = x.view(B, H, num_strips, strip_width, C)
    x = x.permute(0, 2, 1, 3, 4).contiguous()  # B, num_strips, H, strip_width, C
    strips = x.view(-1, H, strip_width, C)
    return strips


def vertical_strip_reverse(strips, strip_width, H, W):
    """
    Reverse vertical strip partition.
    
    Args:
        strips: (num_strips*B, H, strip_width, C)
        strip_width (int): width of each strip
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    num_strips = W // strip_width
    B = int(strips.shape[0] / num_strips)
    x = strips.view(B, num_strips, H, strip_width, -1)
    x = x.permute(0, 2, 1, 3, 4).contiguous()  # B, H, num_strips, strip_width, C
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(  # type: ignore
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class VerticalStripAttention(nn.Module):
    r""" Vertical strip based multi-head self attention module with relative position bias.
    
    Computes attention within vertical strips of the feature map, allowing each position
    to attend to all positions within the same vertical strip. This is designed to capture
    wide-range vertical dependencies for limited-angle artifact reduction.
    
    Supports variable input resolution through position bias interpolation.

    Args:
        dim (int): Number of input channels.
        strip_size (tuple[int]): The height and width of the strip (H, strip_width).
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, strip_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.strip_size = strip_size  # (H, strip_width)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        # For vertical strip: 2*H-1 for vertical, 2*strip_width-1 for horizontal
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * strip_size[0] - 1) * (2 * strip_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the strip
        self.register_buffer("relative_position_index", self._get_relative_position_index(strip_size[0], strip_size[1]))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def _get_relative_position_index(self, H, strip_width):
        """Compute relative position index for a given strip size."""
        coords_h = torch.arange(H)
        coords_w = torch.arange(strip_width)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, H, Sw
        coords_flatten = torch.flatten(coords, 1)  # 2, H*Sw
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, H*Sw, H*Sw
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # H*Sw, H*Sw, 2
        relative_coords[:, :, 0] += H - 1  # shift to start from 0
        relative_coords[:, :, 1] += strip_width - 1
        relative_coords[:, :, 0] *= 2 * strip_width - 1
        relative_position_index = relative_coords.sum(-1)  # H*Sw, H*Sw
        return relative_position_index

    def get_relative_position_bias_for_size(self, target_H, strip_width):
        """
        Get relative position bias for a target size, using interpolation if necessary.
        
        Args:
            target_H (int): Target height (may differ from training height)
            strip_width (int): Strip width (should match training)
        
        Returns:
            relative_position_bias: (num_heads, target_H*strip_width, target_H*strip_width)
        """
        train_H = self.strip_size[0]
        train_Sw = self.strip_size[1]
        
        if target_H == train_H and strip_width == train_Sw:
            # Same size as training, use cached index
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(  # type: ignore
                train_H * train_Sw, train_H * train_Sw, -1)
            return relative_position_bias.permute(2, 0, 1).contiguous()
        
        # Need to interpolate the bias table
        # Reshape bias table to 2D spatial format: (2*H-1, 2*Sw-1, num_heads)
        old_bias = self.relative_position_bias_table.view(
            2 * train_H - 1, 2 * train_Sw - 1, self.num_heads)
        
        # Permute to (num_heads, 2*H-1, 2*Sw-1) for interpolation
        old_bias = old_bias.permute(2, 0, 1).unsqueeze(0)  # (1, num_heads, 2*H-1, 2*Sw-1)
        
        # Target relative position table size
        new_size_h = 2 * target_H - 1
        new_size_w = 2 * strip_width - 1
        
        # Bicubic interpolation
        new_bias = F.interpolate(
            old_bias, size=(new_size_h, new_size_w), 
            mode='bicubic', align_corners=False
        )  # (1, num_heads, 2*target_H-1, 2*strip_width-1)
        
        new_bias = new_bias.squeeze(0).permute(1, 2, 0)  # (2*target_H-1, 2*strip_width-1, num_heads)
        new_bias = new_bias.reshape(-1, self.num_heads)  # ((2*target_H-1)*(2*strip_width-1), num_heads)
        
        # Compute new relative position index for target size
        relative_position_index = self._get_relative_position_index(target_H, strip_width).to(new_bias.device)
        
        # Get bias for target size
        relative_position_bias = new_bias[relative_position_index.view(-1)].view(
            target_H * strip_width, target_H * strip_width, -1)
        
        return relative_position_bias.permute(2, 0, 1).contiguous()

    def forward(self, x, mask=None, actual_H=None):
        """
        Args:
            x: input features with shape of (num_strips*B, N, C) where N = H * strip_width
            mask: (0/-inf) mask with shape of (num_strips, H*Sw, H*Sw) or None
            actual_H: actual height of the input (for variable resolution inference)
        """
        B_, N, C = x.shape
        
        # Determine actual height
        strip_width = self.strip_size[1]
        if actual_H is None:
            actual_H = N // strip_width
        
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Get relative position bias (with interpolation if needed)
        relative_position_bias = self.get_relative_position_bias_for_size(actual_H, strip_width)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nS = mask.shape[0]
            attn = attn.view(B_ // nS, nS, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, strip_size={self.strip_size}, num_heads={self.num_heads}'


class StripTransformerBlock(nn.Module):
    r""" Strip Transformer Block combining local window attention and vertical strip attention.

    This block splits the input channels into two halves:
    - First half (C/2): Local window attention for fine-grained local details
    - Second half (C/2): Vertical strip attention for wide-range vertical dependencies

    The outputs are concatenated back together.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (H, W).
        num_heads (int): Number of attention heads.
        window_size (int): Window size for local attention.
        strip_width (int): Strip width for vertical attention.
        shift_size (int): Shift size for SW-MSA.
        strip_shift_size (int): Shift size for Shifted-Strip-MSA (horizontal direction).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, strip_width=1, shift_size=0,
                 strip_shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.strip_width = strip_width
        self.shift_size = shift_size
        self.strip_shift_size = strip_shift_size
        self.mlp_ratio = mlp_ratio
        
        # Split channels: half for window attention, half for strip attention
        self.dim_window = dim // 2
        self.dim_strip = dim - self.dim_window  # Handle odd dimensions
        self.num_heads_window = num_heads // 2
        self.num_heads_strip = num_heads - self.num_heads_window
        
        # Validate that dimensions are divisible by number of heads
        assert self.num_heads_window > 0 and self.num_heads_strip > 0, \
            f"num_heads ({num_heads}) must be >= 2 to split between window and strip attention"
        assert self.dim_window % self.num_heads_window == 0, \
            f"dim_window ({self.dim_window}) must be divisible by num_heads_window ({self.num_heads_window})"
        assert self.dim_strip % self.num_heads_strip == 0, \
            f"dim_strip ({self.dim_strip}) must be divisible by num_heads_strip ({self.num_heads_strip})"
        
        H, W = input_resolution
        
        # Adjust window size if necessary
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        # Adjust strip width if necessary
        if W <= self.strip_width:
            self.strip_width = W
            self.strip_shift_size = 0
        if self.strip_shift_size > 0:
            assert 0 < self.strip_shift_size < self.strip_width, \
                f"strip_shift_size must be in (0, strip_width), got {self.strip_shift_size}"
        
        # Info message for strip_width=1
        if self.strip_width == 1:
            import warnings
            warnings.warn(
                f"strip_width=1: Pure vertical attention with no horizontal context in strips. "
                f"This is suitable for strong vertical artifacts (e.g., limited-angle CT). "
                f"Consider strip_width >= 2 if you need some horizontal context.",
                UserWarning
            )

        self.norm1 = norm_layer(dim)
        
        # Window attention for first half of channels
        self.attn_window = WindowAttention(
            self.dim_window, window_size=to_2tuple(self.window_size), num_heads=self.num_heads_window,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # Vertical strip attention for second half of channels
        self.attn_strip = VerticalStripAttention(
            self.dim_strip, strip_size=(H, self.strip_width), num_heads=self.num_heads_strip,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Calculate attention masks for shifted window attention
        if self.shift_size > 0:
            attn_mask = self.calculate_window_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

        # Calculate attention masks for shifted strip attention
        if self.strip_shift_size > 0:
            strip_attn_mask = self.calculate_strip_mask(self.input_resolution)
        else:
            strip_attn_mask = None
        self.register_buffer("strip_attn_mask", strip_attn_mask)

    def calculate_window_mask(self, x_size):
        """Calculate attention mask for SW-MSA."""
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def calculate_strip_mask(self, x_size):
        """Calculate attention mask for Shifted-Strip-MSA."""
        H, W = x_size
        
        # Boundary check to avoid overlapping or empty slices
        assert W >= 2 * self.strip_width, \
            f"Width ({W}) must be >= 2 * strip_width ({2 * self.strip_width}). " \
            f"Consider using smaller strip_width or larger input images."
        
        img_mask = torch.zeros((1, H, W, 1))
        
        # Define horizontal slices for shifted strip partitioning (vertical strips)
        w_slices = (
            slice(0, -self.strip_width),
            slice(-self.strip_width, -self.strip_shift_size),
            slice(-self.strip_shift_size, None)
        )
        
        cnt = 0
        for w in w_slices:
            img_mask[:, :, w, :] = cnt
            cnt += 1
        
        # Partition into vertical strips
        mask_strips = vertical_strip_partition(img_mask, self.strip_width)
        mask_strips = mask_strips.view(-1, H * self.strip_width)
        attn_mask = mask_strips.unsqueeze(1) - mask_strips.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Split channels
        x_window = x[..., :self.dim_window]  # B, H, W, C/2
        x_strip = x[..., self.dim_window:]   # B, H, W, C/2

        # ========== Window Attention Branch ==========
        # Cyclic shift for window attention
        if self.shift_size > 0:
            shifted_x_window = torch.roll(x_window, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x_window = x_window

        # Partition windows
        x_windows = window_partition(shifted_x_window, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, self.dim_window)

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            attn_windows = self.attn_window(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn_window(x_windows, mask=self.calculate_window_mask(x_size).to(x.device))

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim_window)
        shifted_x_window = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_window_out = torch.roll(shifted_x_window, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_window_out = shifted_x_window

        # ========== Vertical Strip Attention Branch ==========
        # Cyclic shift for strip attention (horizontal direction only)
        if self.strip_shift_size > 0:
            shifted_x_strip = torch.roll(x_strip, shifts=-self.strip_shift_size, dims=2)
        else:
            shifted_x_strip = x_strip

        # Partition into vertical strips
        x_strips = vertical_strip_partition(shifted_x_strip, self.strip_width)
        x_strips = x_strips.view(-1, H * self.strip_width, self.dim_strip)

        # Shifted-Strip-MSA with mask
        if self.strip_shift_size > 0:
            if self.input_resolution == x_size:
                strip_mask = self.strip_attn_mask
            else:
                strip_mask = self.calculate_strip_mask(x_size).to(x.device)
            attn_strips = self.attn_strip(x_strips, mask=strip_mask, actual_H=H)
        else:
            attn_strips = self.attn_strip(x_strips, mask=None, actual_H=H)

        # Reverse partition
        attn_strips = attn_strips.view(-1, H, self.strip_width, self.dim_strip)
        shifted_x_strip = vertical_strip_reverse(attn_strips, self.strip_width, H, W)

        # Reverse cyclic shift
        if self.strip_shift_size > 0:
            x_strip_out = torch.roll(shifted_x_strip, shifts=self.strip_shift_size, dims=2)
        else:
            x_strip_out = shifted_x_strip

        # ========== Concatenate and Continue ==========
        x = torch.cat([x_window_out, x_strip_out], dim=-1)  # B, H, W, C
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, strip_width={self.strip_width}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # Window attention
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn_window.flops(self.window_size * self.window_size)
        # Strip attention
        nS = W / self.strip_width
        flops += nS * self.dim_strip * 3 * self.dim_strip * H * self.strip_width  # qkv
        flops += self.num_heads_strip * (H * self.strip_width) * (self.dim_strip // self.num_heads_strip) * (H * self.strip_width)  # attn
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class StripBasicLayer(nn.Module):
    """ A basic Swin Transformer layer with Strip attention for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        strip_width (int): Vertical strip width.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, strip_width=1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            StripTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 strip_width=strip_width,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 strip_shift_size=0 if (i % 2 == 0) else strip_width // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()  # type: ignore
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # type: ignore
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim  # type: ignore
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # type: ignore
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class RSTBStrip(nn.Module):
    """Residual Swin Transformer Block with Strip Attention (RSTBStrip).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        strip_width (int): Vertical strip width.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, strip_width=1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTBStrip, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = StripBasicLayer(dim=dim,
                                              input_resolution=input_resolution,
                                              depth=depth,
                                              num_heads=num_heads,
                                              window_size=window_size,
                                              strip_width=strip_width,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop, attn_drop=attn_drop,
                                              drop_path=drop_path,
                                              norm_layer=norm_layer,
                                              downsample=downsample,
                                              use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
        num_out_ch (int): Channel number of output features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        super(UpsampleOneStep, self).__init__()
        self.scale = scale
        self.num_feat = num_feat
        self.num_out_ch = num_out_ch
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution  # type: ignore
        flops = H * W * self.num_feat * 3 * 9
        return flops


class SwinIRStrip(nn.Module):
    r""" SwinIR-Strip
        A PyTorch impl of SwinIR with Strip Attention for image restoration,
        combining local window attention and vertical strip attention.
        Designed for limited-angle CT artifact reduction.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        strip_width (int): Vertical strip width. Default: 1
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, strip_width=1, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinIRStrip, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.strip_width = strip_width

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks with Strip Attention (RSTBStrip)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTBStrip(dim=embed_dim,
                              input_resolution=(patches_resolution[0],
                                                patches_resolution[1]),
                              depth=depths[i_layer],
                              num_heads=num_heads[i_layer],
                              window_size=window_size,
                              strip_width=strip_width,
                              mlp_ratio=self.mlp_ratio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate,
                              drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # type: ignore
                              norm_layer=norm_layer,
                              downsample=None,
                              use_checkpoint=use_checkpoint,
                              img_size=img_size,
                              patch_size=patch_size,
                              resi_connection=resi_connection
                              )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore()
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore()
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        
        # Height only needs to be divisible by window_size (Vertical Strip Attention handles full height defined at init)
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        
        # Calculate LCM of window_size and strip_width for Width padding to ensure divisibility by both
        def lcm(a, b):
            return abs(a * b) // math.gcd(a, b)
            
        mod_w = lcm(self.window_size, self.strip_width)
        mod_pad_w = (mod_w - w % mod_w) % mod_w
        
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution  # type: ignore
        flops += H * W * 3 * self.embed_dim * 9  # type: ignore
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()  # type: ignore
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()  # type: ignore
        return flops


if __name__ == '__main__':
    upscale = 1
    window_size = 8
    strip_width = 1
    height = 64
    width = 64
    
    # Test for image denoising (upsampler='')
    model = SwinIRStrip(
        img_size=(height, width),  # type: ignore
        in_chans=1,
        window_size=window_size,
        strip_width=strip_width,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='',
        upscale=1
    )
    print(model)
    print(f'Image size: {height}x{width}, Window size: {window_size}, Strip width: {strip_width}')
    
    x = torch.randn((1, 1, height, width))
    y = model(x)
    print(f'Input shape: {x.shape}, Output shape: {y.shape}')
