import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple

# ==============================================================
# 1. COMPLEX LINEAR LAYER (with weight initialization)
# ==============================================================

class ComplexLinear(nn.Module):
    """Complex-valued linear transformation preserving phase information."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Xavier initialization for complex weights
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.Wr = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.Wi = nn.Parameter(torch.randn(out_features, in_features) * scale)
        
        self.bias = bias
        if bias:
            self.br = nn.Parameter(torch.zeros(out_features))
            self.bi = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('br', None)
            self.register_parameter('bi', None)

    def forward(self, x):
        """Complex matrix multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i"""
        xr, xi = x.real, x.imag
        
        if self.bias:
            real = F.linear(xr, self.Wr, self.br) - F.linear(xi, self.Wi)
            imag = F.linear(xr, self.Wi, self.bi) + F.linear(xi, self.Wr)
        else:
            real = F.linear(xr, self.Wr) - F.linear(xi, self.Wi)
            imag = F.linear(xr, self.Wi) + F.linear(xi, self.Wr)
        
        return torch.complex(real, imag)


# ==============================================================
# 2. COMPLEX LAYER NORMALIZATION
# ==============================================================

class ComplexLayerNorm(nn.Module):
    """Normalize complex tensors while preserving phase relationships."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        # Normalize magnitude, preserve phase
        magnitude = x.abs()
        phase = torch.angle(x)
        
        mean_mag = magnitude.mean(dim=-1, keepdim=True)
        var_mag = magnitude.var(dim=-1, keepdim=True, unbiased=False)
        
        normalized_mag = (magnitude - mean_mag) / torch.sqrt(var_mag + self.eps)
        normalized_mag = normalized_mag * self.gamma + self.beta
        
        return normalized_mag * torch.exp(1j * phase)


# ==============================================================
# 3. COMPLEX DROPOUT
# ==============================================================

class ComplexDropout(nn.Module):
    """Dropout for complex tensors - same mask for real and imaginary."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x.real) > self.p).float() / (1 - self.p)
        return torch.complex(x.real * mask, x.imag * mask)


# ==============================================================
# 4. HERMITIAN MATRIX (for physical Hamiltonians)
# ==============================================================

class HermitianMatrix(nn.Module):
    """Learnable Hermitian matrix: H = H†"""
    def __init__(self, dim):
        super().__init__()
        # Store as real symmetric + imaginary antisymmetric
        self.Hr = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.Hi = nn.Parameter(torch.randn(dim, dim) * 0.02)
    
    def forward(self):
        # Enforce Hermitian property: H = (A + A^T)/2 + i(B - B^T)/2
        Hr_sym = (self.Hr + self.Hr.T) / 2
        Hi_antisym = (self.Hi - self.Hi.T) / 2
        return torch.complex(Hr_sym, Hi_antisym)


# ==============================================================
# 5. QUANTUM FOURIER ATTENTION (Enhanced)
# ==============================================================

class QuantumFourierAttention(nn.Module):
    """Attention via Fourier-space interference patterns."""
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        
        self.dim = dim
        self.heads = heads
        self.dh = dim // heads
        self.scale = 1 / math.sqrt(self.dh)

        self.to_q = ComplexLinear(dim, dim)
        self.to_k = ComplexLinear(dim, dim)
        self.to_v = ComplexLinear(dim, dim)
        self.to_out = ComplexLinear(dim, dim)
        
        self.norm_q = ComplexLayerNorm(dim)
        self.norm_k = ComplexLayerNorm(dim)
        self.norm_v = ComplexLayerNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = ComplexDropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B, N, D = x.shape

        # Generate QKV with normalization
        Q = self.norm_q(self.to_q(x)).view(B, N, self.heads, self.dh).transpose(1, 2)
        K = self.norm_k(self.to_k(x)).view(B, N, self.heads, self.dh).transpose(1, 2)
        V = self.norm_v(self.to_v(x)).view(B, N, self.heads, self.dh).transpose(1, 2)

        # Transform to Fourier space
        Qf = torch.fft.fft(Q, dim=2, norm='ortho')
        Kf = torch.fft.fft(K, dim=2, norm='ortho')
        Vf = torch.fft.fft(V, dim=2, norm='ortho')

        # Interference: Q * K† in frequency domain
        scores = torch.einsum("bhnd,bhmd->bhnm", Qf, Kf.conj()) * self.scale

        # Softmax on real part (interference intensity)
        attn = F.softmax(scores.real, dim=-1)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, 0)
        
        attn = self.attn_dropout(attn)
        attn = attn.to(torch.complex64)

        # Apply attention to V in frequency space
        out_f = torch.einsum("bhnm,bhmd->bhnd", attn, Vf)

        # Return to time domain
        out = torch.fft.ifft(out_f, dim=2, norm='ortho')

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_dropout(self.to_out(out))


# ==============================================================
# 6. COMPLEX FEEDFORWARD (Enhanced with GELU activation)
# ==============================================================

class QuantumFeedForward(nn.Module):
    """Complex feedforward with proper nonlinear activation."""
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        hidden = dim * mult
        self.lin1 = ComplexLinear(dim, hidden)
        self.lin2 = ComplexLinear(hidden, dim)
        self.drop1 = ComplexDropout(dropout)
        self.drop2 = ComplexDropout(dropout)
        self.norm = ComplexLayerNorm(hidden)

    def forward(self, x):
        # Apply complex GELU: separate on magnitude, preserve phase structure
        x = self.lin1(x)
        x = self.norm(x)
        
        # Complex activation: apply GELU to magnitude, preserve phase
        magnitude = x.abs()
        phase = torch.angle(x)
        magnitude = F.gelu(magnitude)
        x = magnitude * torch.exp(1j * phase)
        
        x = self.drop1(x)
        x = self.lin2(x)
        return self.drop2(x)


# ==============================================================
# 7. SCHRÖDINGER EVOLUTION (Enhanced with stability)
# ==============================================================

class SchrodingerEvolution(nn.Module):
    """Unitary time evolution via Schrödinger equation."""
    def __init__(self, dim, dt=0.05, max_dt=0.2):
        super().__init__()
        self.H = HermitianMatrix(dim)
        # Learnable time step with bounds
        self.dt = nn.Parameter(torch.tensor(dt))
        self.max_dt = max_dt

    def forward(self, x):
        B, N, D = x.shape
        
        # Clamp dt for stability
        dt = torch.clamp(self.dt, 0.01, self.max_dt)
        
        # Get Hermitian Hamiltonian
        H = self.H()
        
        # Compute unitary evolution operator: U = exp(-iHt)
        # Use eigendecomposition for numerical stability
        x_flat = x.reshape(B * N, D)
        
        # For efficiency, use matrix exponential
        U = torch.matrix_exp(-1j * H * dt)
        
        # Apply evolution
        out = torch.matmul(x_flat, U)
        
        # Normalize to preserve norm (unitary evolution)
        norm = out.abs().mean(dim=-1, keepdim=True) + 1e-8
        out = out / norm
        
        return out.reshape(B, N, D)


# ==============================================================
# 8. WAVE TRANSFORMER BLOCK (Enhanced with residuals)
# ==============================================================

class WaveTransformerBlock(nn.Module):
    """Complete transformer block with wave evolution."""
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.attn = QuantumFourierAttention(dim, heads, dropout)
        self.ff = QuantumFeedForward(dim, 4, dropout)
        self.sch = SchrodingerEvolution(dim)
        
        self.norm1 = ComplexLayerNorm(dim)
        self.norm2 = ComplexLayerNorm(dim)
        self.norm3 = ComplexLayerNorm(dim)
        
        self.drop = ComplexDropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Attention with pre-norm and residual
        x = x + self.drop(self.attn(self.norm1(x), mask))
        
        # Feedforward with pre-norm and residual
        x = x + self.drop(self.ff(self.norm2(x)))
        
        # Schrödinger evolution (unitary transformation)
        x = self.sch(self.norm3(x))
        
        return x


# ==============================================================
# 9. FULL QUANTUM-WAVE TRANSFORMER (Enhanced)
# ==============================================================

class QuantumWaveTransformer(nn.Module):
    """Complete wave-based transformer architecture."""
    def __init__(self, dim=64, depth=8, heads=8, dropout=0.1, 
                 input_dim=None, output_dim=None):
        super().__init__()
        
        self.dim = dim
        self.input_dim = input_dim or dim
        self.output_dim = output_dim or dim
        
        # Input projection to complex space
        self.input_proj = ComplexLinear(self.input_dim, dim)
        self.input_norm = ComplexLayerNorm(dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            WaveTransformerBlock(dim, heads, dropout) 
            for _ in range(depth)
        ])
        
        # Output projection back to real space
        self.output_norm = ComplexLayerNorm(dim)
        self.output_proj = nn.Linear(dim * 2, self.output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize output projection."""
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Convert to complex with zero imaginary part
        x = torch.complex(x, torch.zeros_like(x))
        
        # Project to complex hidden space
        x = self.input_norm(self.input_proj(x))
        
        # Pass through transformer blocks
        for blk in self.blocks:
            x = blk(x, mask)
        
        # Normalize output
        x = self.output_norm(x)
        
        # Convert back to real: concatenate real and imaginary parts
        x_real = torch.cat([x.real, x.imag], dim=-1)
        
        return self.output_proj(x_real)


# ==============================================================
# 10. ENHANCED PHYSICS DATA GENERATORS
# ==============================================================

def generate_rotation(batch, length=64, omega=1.0):
    """Generate rotating wave packet."""
    t = torch.linspace(0, 4 * math.pi, length)
    x = torch.sin(omega * t)
    y = torch.cos(omega * t)
    
    # Add noise for robustness
    noise = torch.randn_like(x) * 0.01
    x = x + noise
    y = y + noise
    
    base = torch.stack([x, y], dim=-1)
    base = base.unsqueeze(0).repeat(batch, 1, 1)
    return torch.cat([base, torch.zeros(batch, length, 62)], dim=-1)


def generate_RLC(batch, length=64, alpha=0.3, omega=1.5):
    """Generate damped oscillator (RLC circuit)."""
    t = torch.linspace(0, 10, length)
    
    # Damped oscillation with varying parameters
    alphas = alpha + torch.randn(batch, 1) * 0.05
    omegas = omega + torch.randn(batch, 1) * 0.1
    
    I = torch.exp(-alphas * t) * torch.cos(omegas * t)
    
    # Normalize
    I = I / (I.abs().max(dim=-1, keepdim=True)[0] + 1e-6)
    
    # Current and voltage (derivative)
    dI = torch.gradient(I, dim=-1)[0]
    
    base = torch.stack([I, dI], dim=-1)
    return torch.cat([base, torch.zeros(batch, length, 62)], dim=-1)


def generate_coupled_oscillators(batch, length=64):
    """Generate coupled harmonic oscillators."""
    t = torch.linspace(0, 10, length)
    
    # Coupling parameter
    k = 0.5
    
    x1 = torch.cos(t) + 0.5 * torch.cos(k * t)
    x2 = torch.cos(t) - 0.5 * torch.cos(k * t)
    
    base = torch.stack([x1, x2], dim=-1).unsqueeze(0).repeat(batch, 1, 1)
    return torch.cat([base, torch.zeros(batch, length, 62)], dim=-1)


# ==============================================================
# 11. ENHANCED SCHRÖDINGER DATASET
# ==============================================================

def schrodinger_dataset(batch=4, T=64, N=256, dt=0.01, 
                       potential_type='harmonic'):
    """
    Generate quantum wavefunction evolution using split-step method.
    
    Args:
        potential_type: 'harmonic', 'double_well', 'barrier'
    """
    x = np.linspace(-10, 10, N)
    dx = x[1] - x[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    
    # Define potential
    if potential_type == 'harmonic':
        V = 0.5 * x**2
    elif potential_type == 'double_well':
        V = 0.1 * (x**2 - 5)**2
    elif potential_type == 'barrier':
        V = np.where(np.abs(x) < 2, 2.0, 0.0)
    else:
        V = 0.5 * x**2
    
    all_sequences = []
    
    for _ in range(batch):
        # Random initial wave packet
        x0 = np.random.uniform(-3, 3)
        k0 = np.random.uniform(0.5, 3.0)
        sigma = np.random.uniform(0.5, 1.5)
        
        psi = np.exp(-((x - x0)**2) / (2 * sigma**2)) * np.exp(1j * k0 * x)
        psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)
        
        seq = []
        
        for _ in range(T):
            seq.append(psi.copy())
            
            # Split-step Fourier method
            psi *= np.exp(-1j * V * dt / 2)
            psi_k = np.fft.fft(psi)
            psi_k *= np.exp(-1j * (k**2) / 2 * dt)
            psi = np.fft.ifft(psi_k)
            psi *= np.exp(-1j * V * dt / 2)
            
            # Renormalize
            norm = np.sqrt(np.sum(np.abs(psi)**2) * dx)
            psi /= norm
        
        all_sequences.append(np.stack(seq))
    
    return torch.from_numpy(np.stack(all_sequences)).to(torch.complex64)


# ==============================================================
# 12. ENHANCED DATA PREPARATION
# ==============================================================

def prepare_schrodinger_batch(batch=4, T=64, compress_to=32):
    """Compress wavefunction to Fourier modes for transformer input."""
    psi = schrodinger_dataset(batch, T=T)  # (B, T, N)
    
    # FFT compression
    psi_fft = torch.fft.fft(psi, dim=-1, norm='ortho')
    psi_modes = psi_fft[..., :compress_to]
    
    # Convert to real representation
    psi_real = torch.cat([psi_modes.real, psi_modes.imag], dim=-1)
    
    return psi_real.float(), psi_real.float()


# ==============================================================
# 13. ENHANCED TRAINING WITH METRICS
# ==============================================================

def train(model, generator, steps=800, lr=1e-4, log_interval=100):
    """Train with learning rate scheduling and metrics."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    
    losses = []
    
    for step in range(steps):
        x = generator(8).float()
        pred = model(x)
        
        loss = F.mse_loss(pred, x)
        
        # L1 regularization for sparsity
        l1_loss = sum(p.abs().sum() for p in model.parameters()) * 1e-6
        total_loss = loss + l1_loss
        
        opt.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if step % log_interval == 0:
            avg_loss = np.mean(losses[-log_interval:]) if len(losses) >= log_interval else np.mean(losses)
            print(f"[{step:4d}] Loss: {loss.item():.6f} | Avg: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
    
    return model, losses


def train_schrodinger(model, steps=1500, lr=1e-4):
    """Enhanced training for Schrödinger equation."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr*10, steps_per_epoch=steps, epochs=1
    )
    
    losses = []
    
    for step in range(steps):
        x, target = prepare_schrodinger_batch(4)
        pred = model(x)
        
        loss = F.mse_loss(pred, target)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"[Schrodinger {step:4d}] Loss: {loss.item():.6f}")
    
    return losses


# ==============================================================
# 14. ENHANCED VISUALIZATION
# ==============================================================

def visualize(model, generator, save_path=None):
    """Enhanced visualization with multiple samples."""
    model.eval()
    with torch.no_grad():
        x = generator(4).float()
        pred = model(x).detach()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(4):
        gt = x[i, :, 0].numpy()
        pr = pred[i, :, 0].numpy()
        
        axes[i].plot(gt, label='Ground Truth', linewidth=2, alpha=0.7)
        axes[i].plot(pr, label='Prediction', linewidth=2, alpha=0.7)
        axes[i].legend()
        axes[i].set_title(f'Sample {i+1}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    model.train()


def visualize_schrodinger(model, save_path=None):
    """Enhanced Schrödinger visualization with probability density."""
    model.eval()
    with torch.no_grad():
        x, _ = prepare_schrodinger_batch(1)
        pred = model(x).detach()
    
    # Reconstruct wavefunction
    real = pred[0, :, :32]
    imag = pred[0, :, 32:]
    psi_fft = torch.complex(real, imag)
    
    psi = torch.fft.ifft(psi_fft, n=256, dim=-1, norm='ortho')
    prob = psi.abs().cpu().numpy()**2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Probability density evolution
    im1 = ax1.imshow(prob, aspect='auto', cmap='plasma', origin='lower')
    ax1.set_title('|ψ(x,t)|² - Quantum Wavefunction Evolution')
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel('Time (t)')
    plt.colorbar(im1, ax=ax1, label='Probability Density')
    
    # Snapshots at different times
    times = [0, len(prob)//3, 2*len(prob)//3, -1]
    for t_idx in times:
        ax2.plot(prob[t_idx], label=f't={t_idx}', alpha=0.7)
    
    ax2.set_title('Wavefunction Snapshots')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('|ψ|²')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    model.train()


def plot_training_curves(losses, title='Training Loss', save_path=None):
    """Plot training loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, alpha=0.6, label='Loss')
    
    # Moving average
    window = min(50, len(losses) // 10)
    if window > 1:
        ma = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), ma, 
                linewidth=2, label=f'MA({window})')
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


# ==============================================================
# 15. MODEL ANALYSIS TOOLS
# ==============================================================

def analyze_model(model):
    """Print model statistics and parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Model: {model.__class__.__name__}")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"{'='*60}\n")
    
    # Layer breakdown
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:20s}: {params:>10,} params")


# ==============================================================
# MAIN EXECUTION
# ==============================================================

def main():
    print("\n" + "="*70)
    print(" QuantumWave Transformer v0.2 - Enhanced Implementation")
    print("="*70 + "\n")
    
    # Model configuration
    model = QuantumWaveTransformer(
        dim=64,
        depth=8,
        heads=8,
        dropout=0.1
    )
    
    analyze_model(model)
    
    # Experiment 1: Rotation
    print("\n[Experiment 1] Training on Circular Rotation...")
    print("-" * 70)
    model_rot, losses_rot = train(model, generate_rotation, steps=800)
    plot_training_curves(losses_rot, 'Rotation Training Loss')
    visualize(model_rot, generate_rotation, 'rotation_results.png')
    
    # Experiment 2: RLC Circuit
    print("\n[Experiment 2] Training on Damped Oscillator (RLC)...")
    print("-" * 70)
    model_rlc = QuantumWaveTransformer(dim=64, depth=8, heads=8)
    model_rlc, losses_rlc = train(model_rlc, generate_RLC, steps=800)
    plot_training_curves(losses_rlc, 'RLC Circuit Training Loss')
    visualize(model_rlc, generate_RLC, 'rlc_results.png')
    
    # Experiment 3: Coupled Oscillators
    print("\n[Experiment 3] Training on Coupled Oscillators...")
    print("-" * 70)
    model_coupled = QuantumWaveTransformer(dim=64, depth=8, heads=8)
    model_coupled, losses_coupled = train(model_coupled, generate_coupled_oscillators, steps=800)
    visualize(model_coupled, generate_coupled_oscillators, 'coupled_results.png')
    
    # Experiment 4: Schrödinger Equation
    print("\n[Experiment 4] Training on Quantum Wave Equation...")
    print("-" * 70)
    model_quantum = QuantumWaveTransformer(dim=64, depth=8, heads=8)
    losses_sch = train_schrodinger(model_quantum, steps=1500)
    plot_training_curves(losses_sch, 'Schrödinger Training Loss')
    visualize_schrodinger(model_quantum, 'schrodinger_results.png')
    
    print("\n" + "="*70)
    print(" Training Complete - All experiments finished successfully")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()