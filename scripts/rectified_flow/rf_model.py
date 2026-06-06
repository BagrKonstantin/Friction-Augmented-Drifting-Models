from typing import Tuple

import numpy as np
import torch 
 
import torch.nn as nn
import string
import torch.nn.functional as F
from tqdm import tqdm
import scripts.rectified_flow.simple_updown_sampling as up_or_down_sampling

import torch.nn as nn
import torch


class LatentRFUNet(nn.Module):
    """Wraps RFUNet to operate on flat latent vectors [B, latent_dim]."""

    def __init__(self, latent_dim=512, spatial_ch=2, spatial_size=16):
        super().__init__()
        assert spatial_ch * spatial_size * spatial_size == latent_dim, \
            f"spatial_ch * spatial_size^2 must equal latent_dim ({latent_dim})"
        self.spatial_ch = spatial_ch
        self.spatial_size = spatial_size
        self.net = RFUNet(in_channels=spatial_ch)

    def forward(self, z, t):
        B = z.shape[0]
        if t.dim() == 2:
            t = t.squeeze(1)                    # [B]
        t_scaled = (t * 999).long()             # float [0,1] → int [0,999]
        x = z.view(B, self.spatial_ch, self.spatial_size, self.spatial_size)
        out = self.net(x, t_scaled)
        return out.view(B, -1)


class TimeConditionedMLP(nn.Module):
    """
    Time-conditioned MLP for Rectified Flow velocity field v(z, t).
    Architecture from Rectified Flow paper for latent space transformation.
    
    Input: z (latent) concatenated with t (time)
    Output: velocity field v(z,t) with same dimension as z
    """
    def __init__(self, dim=512, hidden_dim=1024, num_layers=3, activation='tanh'):
        super().__init__()
        self.dim = dim
        
        layers = []
        # First layer: (dim + 1) -> hidden_dim (the +1 is for time t)
        layers.append(nn.Linear(dim + 1, hidden_dim))
        
        if activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'relu':
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        layers.append(act_fn())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())
        
        # Output layer: hidden_dim -> dim
        layers.append(nn.Linear(hidden_dim, dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, z, t):
        """
        Args:
            z: [batch_size, dim] - latent vectors
            t: [batch_size, 1] or [batch_size] - time values in [0, 1]
        Returns:
            velocity: [batch_size, dim] - velocity field v(z,t)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [batch_size, 1]
        
        # Concatenate z and t
        zt = torch.cat([z, t], dim=1)  # [batch_size, dim + 1]
        
        # Predict velocity
        velocity = self.net(zt)
        
        return velocity


class RectifiedFlow:
    """
    Rectified Flow model for learning optimal transport between two distributions.
    
    The key idea: learn velocity field v(z,t) such that solving ODE:
        dZ/dt = v(Z_t, t)
    transports samples from π₀ to π₁.
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def get_train_tuple(self, z0: torch.tensor, z1: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Generate training data for rectified flow.
        
        Sample random time t ∈ [0,1], compute:
        - z_t = t·z1 + (1-t)·z0  (linear interpolation)
        - target = z1 - z0  (velocity of straight line)
        
        Args:
            z0: [batch_size, dim] - source samples
            z1: [batch_size, dim] - target samples
        Returns:
            z_t: [batch_size, dim] - interpolated samples
            t: [batch_size, 1] - random times
            target: [batch_size, dim] - target velocity
        """
        batch_size = z0.shape[0]
        
        # Sample random time t ∈ [0, 1]
        t = torch.rand((batch_size, 1), device=z0.device)
        
        # Linear interpolation: z_t = t·z1 + (1-t)·z0
        z_t = t * z1 + (1 - t) * z0
        
        # Target velocity (constant along straight line)
        target = z1 - z0
        
        return z_t, t, target
    
    @torch.no_grad()
    def sample_ode(self, z0, num_steps=100, mute_tqdm=False):
        """
        Sample from the learned flow using Euler method.
        
        Solve ODE: dZ/dt = v(Z_t, t) from t=0 to t=1
        
        Args:
            z0: [batch_size, dim] - initial samples from π₀
            num_steps: number of discretization steps
        Returns:
            z1: [batch_size, dim] - transported samples ≈ samples from π₁
        """
        batch_size = z0.shape[0]
        dt = 1.0 / num_steps
        
        z = z0.clone()
        
        for step in tqdm(range(num_steps), desc="Sampling ODE", disable=mute_tqdm):
            t = torch.ones((batch_size, 1), device=z.device) * (step / num_steps)
            
            # Predict velocity
            velocity = self.model(z, t)
            
            # Euler step
            z = z + velocity * dt
        
        return z


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    """
    Create sinusoidal timestep embeddings.
    
    Args:
        timesteps: 1-D Tensor of N indices, one per batch element.
        embedding_dim: Dimension of the embeddings to generate.
        max_positions: Maximum number of positions.
    
    Returns:
        Tensor of shape (N, embedding_dim)
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = np.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
            "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init


def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return variance_scaling(scale, 'fan_avg', 'uniform')



class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)



def conv1x1(in_planes, out_planes, stride=1, bias=True, init_scale=1., padding=0):
  """1x1 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  if conv.bias is not None:
    nn.init.zeros_(conv.bias)
  return conv

def conv3x3(in_planes, out_planes, stride=1, bias=True, dilation=1, init_scale=1., padding=1):
  """3x3 convolution with DDPM initialization."""
  conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding,
                   dilation=dilation, bias=bias)
  conv.weight.data = default_init(init_scale)(conv.weight.data.shape)
  if conv.bias is not None:
    nn.init.zeros_(conv.bias)
  return conv


class ResnetBlockBigGAN(nn.Module):
    
    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
        super().__init__()

        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.up = up
        self.down = down
        self.fir = fir
        self.fir_kernel = fir_kernel

        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
            nn.init.zeros_(self.Dense_0.bias)

        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch or up or down:
            self.Conv_2 = conv1x1(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.in_ch = in_ch
        self.out_ch = out_ch
      
   
    def forward(self, x, temb=None):
      h = self.act(self.GroupNorm_0(x))

      if self.up:
        if self.fir:
          h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
          x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
        else:
          h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
          x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
      elif self.down:
        if self.fir:
          h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
          x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
        else:
          h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
          x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

      h = self.Conv_0(h)
      # Add bias to each feature map conditioned on the time embedding
      if temb is not None:
        h += self.Dense_0(self.act(temb))[:, :, None, None]
      h = self.act(self.GroupNorm_1(h))
      h = self.Dropout_0(h)
      h = self.Conv_1(h)

      if self.in_ch != self.out_ch or self.up or self.down:
        x = self.Conv_2(x)

      if not self.skip_rescale:
        return x + h
      else:
        return (x + h) / np.sqrt(2.)


def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_lowercase[len(x.shape):len(y.shape) + len(x.shape)])
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
  def __init__(self, in_dim, num_units, init_scale=0.1):
    super().__init__()
    self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
    self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    y = contract_inner(x, self.W) + self.b
    return y.permute(0, 3, 1, 2)


class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0.):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                  eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    self.skip_rescale = skip_rescale

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)



class RFUNet(nn.Module):

      def __init__(self, in_channels=3):
        super(RFUNet, self).__init__()
        self.layers = nn.ModuleList(modules=[
            # --- Time embedding (positional: sinusoidal computed in forward, not a module) ---
            nn.Linear(128, 512),
            Swish(),# embed_dim=nf=128 → nf*4=512
            nn.Linear(512, 512),

            # --- Input conv ---
            conv3x3(in_channels, 128), # [2]

            # === ENCODER ===
            # Level 0 — res 32×32, ch=128  [no attn]
            ResnetBlockBigGAN(in_ch=128, out_ch=128, act=Swish(), temb_dim=512, skip_rescale=True),
            ResnetBlockBigGAN(in_ch=128, out_ch=128, act=Swish(), temb_dim=512),
            ResnetBlockBigGAN(in_ch=128, out_ch=128, act=Swish(), temb_dim=512),
            ResnetBlockBigGAN(in_ch=128, out_ch=128, act=Swish(), temb_dim=512),
            ResnetBlockBigGAN(in_ch=128, out_ch=128, act=Swish(), temb_dim=512, down=True),   # naive avg-pool ×2 (no FIR), shortcut conv1x1

            # Level 1 — res 16×16, ch=256  [attn after every resblock]
            ResnetBlockBigGAN(in_ch=128, out_ch=256, act=Swish(), temb_dim=512),    # shortcut conv1x1(128,256)
            AttnBlockpp(256),                         # GroupNorm+QKV NIN, skip_rescale /√2
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),
            AttnBlockpp(256),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),
            AttnBlockpp(256),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512, down=True),   # naive avg-pool ×2, shortcut conv1x1

            # Level 2 — res 8×8, ch=256  [no attn]
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512, down=True),   # naive avg-pool ×2, shortcut conv1x1

            # Level 3 — res 4×4, ch=256  [no attn, no downsample]
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),

            # === BOTTLENECK ===
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),
            AttnBlockpp(256),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512),

            # === DECODER ===
            # Level 3 — res 4×4  (num_res_blocks+1=5 blocks, cat skip)
            ResnetBlockBigGAN(in_ch=256+256, out_ch=256, act=Swish(), temb_dim=512),      # pop: level3_b3 (256)
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level3_b2
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level3_b1
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level3_b0
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level2_downsample (256)
            # no attn at res=4
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512, up=True),     # naive nearest ×2, shortcut conv1x1

            # Level 2 — res 8×8
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level2_b3
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level2_b2
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level2_b1
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level2_b0
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level1_downsample (256)
            # no attn at res=8
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512, up=True),     # naive nearest ×2, shortcut conv1x1
            # Level 1 — res 16×16  [attn after all 5 resblocks]
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level1_b3 (256)
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level1_b2
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level1_b1
            ResnetBlockBigGAN(in_ch=512, out_ch=256, act=Swish(), temb_dim=512),              # pop: level1_b0
            ResnetBlockBigGAN(in_ch=256+128, out_ch=256, act=Swish(), temb_dim=512),      # pop: level0_downsample (128) ← channel mismatch
            AttnBlockpp(256),
            ResnetBlockBigGAN(in_ch=256, out_ch=256, act=Swish(), temb_dim=512, up=True),     # naive nearest ×2, shortcut conv1x1

            # Level 0 — res 32×32  [no attn, no upsample]
            ResnetBlockBigGAN(in_ch=256+128, out_ch=128, act=Swish(), temb_dim=512),      # pop: level0_b3 (128), shortcut conv1x1(384,128)
            ResnetBlockBigGAN(in_ch=128+128, out_ch=128, act=Swish(), temb_dim=512),      # pop: level0_b2
            ResnetBlockBigGAN(in_ch=256, out_ch=128, act=Swish(), temb_dim=512),              # pop: level0_b1
            ResnetBlockBigGAN(in_ch=256, out_ch=128, act=Swish(), temb_dim=512),              # pop: level0_b0
            ResnetBlockBigGAN(in_ch=256, out_ch=128, act=Swish(), temb_dim=512),           # pop: initial conv output (128)

            # === OUTPUT ===
            nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-6),
            conv3x3(128, in_channels, init_scale=0.)
          ]
        )
      def forward(self, x, t):
        temb = get_timestep_embedding(t, 128)
        time_embedding = self.layers[0](temb)
        time_embedding = self.layers[1](time_embedding)
        time_embedding = self.layers[2](time_embedding)

        h = self.layers[3](x)
        skips = [h]                                       # input conv 128-ch 32×32

        # Encoder level 0 (32×32, 128-ch)
        for i in range(4, 8):
          h = self.layers[i](h, time_embedding); skips.append(h)
        h = self.layers[8](h, time_embedding);  skips.append(h)   # down → 16×16

        # Encoder level 1 (16×16, 256-ch) — save b0, attn0, b1, b2 (4 × 256-ch at 16×16)
        h = self.layers[9](h, time_embedding);  skips.append(h)   # b0
        h = self.layers[10](h);                 skips.append(h)   # attn0
        h = self.layers[11](h, time_embedding); skips.append(h)   # b1
        h = self.layers[12](h)                                     # attn1 no save
        h = self.layers[13](h, time_embedding); skips.append(h)   # b2
        h = self.layers[14](h)                                     # attn2 no save
        h = self.layers[15](h, time_embedding); skips.append(h)   # down → 8×8

        # Encoder level 2 (8×8, 256-ch) — save last 4 of 6 blocks to match decoder count
        h = self.layers[16](h, time_embedding)                     # b0 no save
        h = self.layers[17](h, time_embedding)                     # b1 no save
        for i in range(18, 22):
          h = self.layers[i](h, time_embedding); skips.append(h)
        h = self.layers[22](h, time_embedding); skips.append(h)   # down → 4×4

        # Encoder level 3 (4×4, 256-ch)
        for i in range(23, 27):
          h = self.layers[i](h, time_embedding); skips.append(h)

        # Bottleneck
        h = self.layers[27](h, time_embedding)
        h = self.layers[28](h)
        h = self.layers[29](h, time_embedding)

        # Decoder level 3 (4×4)
        for i in range(30, 35):
          h = torch.cat([h, skips.pop()], dim=1)
          h = self.layers[i](h, time_embedding)
        h = self.layers[35](h, time_embedding)   # up → 8×8

        # Decoder level 2 (8×8)
        for i in range(36, 41):
          h = torch.cat([h, skips.pop()], dim=1)
          h = self.layers[i](h, time_embedding)
        h = self.layers[41](h, time_embedding)   # up → 16×16

        # Decoder level 1 (16×16)
        for i in range(42, 47):
          h = torch.cat([h, skips.pop()], dim=1)
          h = self.layers[i](h, time_embedding)
        h = self.layers[47](h)                   # attn
        h = self.layers[48](h, time_embedding)   # up → 32×32

        # Decoder level 0 (32×32)
        for i in range(49, 54):
          h = torch.cat([h, skips.pop()], dim=1)
          h = self.layers[i](h, time_embedding)

        # Output
        h = self.layers[54](h)
        h = self.layers[55](h)
        return h
    


if __name__ == "__main__":
    model = RFUNet()
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 1000, (2,))
    out = model(x, t)
    print(out.shape)