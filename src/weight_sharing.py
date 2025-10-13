import jax
import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Optional
from torch.autograd import Function

from .utils import params_to_vector


class ParameterSharing(nn.Module):
    """
    Base class for parameter sharing methods in neural network optimization.
    
    This class provides a framework for reducing the dimensionality of neural network
    parameter spaces by sharing parameters through various mapping strategies. It enables
    optimization in a lower-dimensional latent space while maintaining the full expressivity
    of the original parameter space.
    
    The general workflow is:
    1. Map from low-dimensional latent space (d dimensions) to full parameter space (D dimensions)
    2. Process the expanded parameters
    3. Load the parameters into the neural network model
    
    Input types:
        - Latent vector z: Union[np.ndarray, torch.Tensor] of shape (d,)
    
    Output types:
        - Expanded parameters: torch.Tensor of shape (D,) representing full model parameters
        - Processed parameters: torch.Tensor of shape (D,) ready for loading into model
    """

    def __init__(self, model: torch.nn.Module, d: int, alpha: float = 1.0, device: str = 'cuda', seed: int = 0) -> None:
        """
        Initialize parameter sharing.

        Args:
            model: PyTorch model (e.g., ResNet-18).
            d: Number of shared parameters (latent dimension K).
            alpha: Scaling factor for parameters.
            device: Device for PyTorch computations.
            seed: Random seed for reproducible initialization.
        """
        super().__init__()
        self.device = device
        self.model = model
        self.seed = seed
        theta_base = params_to_vector(self.model.parameters())
        self.register_buffer('theta_base', theta_base)
        self.D = len(theta_base)
        self.d = d
        self.alpha = alpha

        assignments = torch.from_numpy(np.random.randint(0, self.d, (self.D,)))
        self.register_buffer('assignments', assignments)

        self.x0 = self.init_x0()

    def init_x0(self) -> np.ndarray:
        """Initialize the starting point in latent space.
        
        Returns:
            Initial latent vector of shape (d,)
        """
        return np.zeros(self.d)

    def set_model(self, model: torch.nn.Module) -> None:
        """Update the model and recalculate base parameters.
        
        Args:
            model: New PyTorch model to use
        """
        self.model = model
        self.theta_base = params_to_vector(self.model.parameters())
        self.D = len(self.theta_base)

    def set_theta(self, theta: Union[np.ndarray, torch.Tensor]) -> None:
        """Set the base parameter vector.
        
        Args:
            theta: Base parameter vector of shape (D,)
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.from_numpy(theta)
        self.theta_base = theta.to(self.device)

    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using simple assignment strategy.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Expanded parameter vector of shape (D,)
        """
        z = self._to_tensor(z)
        x = z[self.assignments]
        return self.process(x)
    
    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to torch tensor on correct device."""
        if not isinstance(x, torch.Tensor):
            if hasattr(x, 'shape'):  # JAX array
                x = np.array(x)
            if isinstance(x, jax.Array):
                x = np.array(x)
            x = torch.from_numpy(x).float()
        return x.to(self.device)
    
    def process(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Process expanded parameters with scaling and optional normalization.
        
        Args:
            x: Expanded parameter vector of shape (D,)
            
        Returns:
            Processed parameter tensor of shape (D,)
        """
        x = self._to_tensor(x)
        return self.alpha * x
    
    def load_to_model(self, theta: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Load processed parameters into the neural network model.
        
        Args:
            theta: Parameter tensor of shape (D,) to load into model
        """
        theta = self._to_tensor(theta)
        torch.nn.utils.vector_to_parameters(theta, self.model.parameters())
    

class RandomProjectionSoftSharing(ParameterSharing):
    """
    Parameter sharing using random projection matrices.
    
    This class implements soft parameter sharing by projecting from a low-dimensional
    latent space to the full parameter space using a random projection matrix P.
    The mapping is: theta = theta_base + P @ z, where P is a random matrix of shape (D, d).
    
    The projection matrix can optionally be normalized to have unit column norms
    and scaled by 1/sqrt(d) to maintain reasonable parameter magnitudes.
    
    Input types:
        - Model: torch.nn.Module
        - d: int (number of shared parameters)
        - Scaling factor alpha: float
        - Normalization: bool (whether to normalize the projection matrix)  
        
    
    Output types:
        - Full parameter vector: torch.Tensor of shape (D,) where D is the
          total number of model parameters
    """
    
    def __init__(self, model: torch.nn.Module, d: int, alpha: float = 1.0, normalize: bool = False, device: str = 'cuda', seed: int = 0) -> None:
        super().__init__(model, d, alpha, device, seed)
        self.normalize = normalize
        self.init()

    def init(self) -> torch.Tensor:
        """
        Initialize the random projection matrix with seed.
        
        Returns:
            Random projection matrix P of shape (D, d)
        """
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        P = torch.randn(self.D, self.d, device=self.device, generator=generator)
        if self.normalize:
            # P = P / P.norm(dim=0, keepdim=True)
            Q, _ = torch.linalg.qr(P)
            P = Q
            # P = Q.T
            # P = Q[:, :self.d].T

            # Check for orthonormality: P @ P.T should be the identity matrix
            # print(P.shape, P.T.shape)
            # print(torch.allclose(P @ P.T, torch.eye(self.d, device=self.device)))
            # print(P.T @ P)
            P = P / (self.d ** 0.5)
        self.register_buffer('P', P)
        return P

    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using random projection.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,) computed as theta_base + P @ z
        """
        z = self._to_tensor(z)
        x = self.P @ z
        x = self.process(x)
        return self.theta_base + x

    def latent_rotate(self, R: torch.Tensor) -> None:
        """Apply an orthogonal rotation in latent space: P <- P @ R.
        Args:
            R: Rotation matrix of shape (d, d), should be orthonormal.
        """
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, dtype=self.P.dtype, device=self.device)
        R = R.to(self.device)
        self.P = self.P @ R


class RandomFourierFeaturesSoftSharing(ParameterSharing):
    """
    Parameter sharing using Random Fourier Features (RFF).
    
    This class implements soft parameter sharing by using Random Fourier Features
    to map from a low-dimensional latent space to the full parameter space.
    The mapping follows the form:
    x = theta_base + sqrt(2/D) * cos(omega @ z + b)
    
    where:
    - omega is sampled from N(0, sigma^{-2} I) with shape (D, d)
    - b is sampled uniformly from [0, 2π] with shape (D,)
    - z is the latent vector of shape (d,)
    - D is the total number of model parameters
    
    Random Fourier Features approximate a Gaussian kernel and provide a way
    to capture non-linear relationships in the parameter space through
    trigonometric transformations.
    
    Input types:
        - Latent vector z: Union[np.ndarray, torch.Tensor] of shape (d,)
    
    Output types:
        - Full parameter vector: torch.Tensor of shape (D,) where D is the
          total number of model parameters
    """
    
    def __init__(self, model: torch.nn.Module, d: int, 
                 sigma: float = 1.0, alpha: float = 1.0, device: str = 'cuda', seed: int = 0) -> None:
        """
        Initialize Random Fourier Features soft sharing.
        
        Args:
            model: PyTorch model (e.g., ResNet-18).
            d: Number of shared parameters (latent dimension).
            sigma: Standard deviation for omega sampling (1/sigma^2 is the variance).
            device: Device for PyTorch computations.
            seed: Random seed for reproducible initialization.
        """
        super().__init__(model, d, alpha, device, seed)
        self.sigma = sigma
        self.init()

    def init(self) -> None:
        """
        Initialize Random Fourier Features parameters with seed.
        
        Draws omega from N(0, sigma^{-2} I) and b from uniform [0, 2π].
        """
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        
        # Draw omegas from N(0, sigma^{-2} I)
        omega = torch.randn(self.D, self.d, device=self.device, generator=generator) / self.sigma
        self.register_buffer('omega', omega)
        
        # Draw biases from uniform [0, 2π]
        b = torch.rand(self.D, device=self.device, generator=generator) * 2 * np.pi
        self.register_buffer('b', b)

    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using Random Fourier Features.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,) computed as:
            theta_base + sqrt(2/D) * cos(omega @ z + b)
        """
        z = self._to_tensor(z)
        linear_combination = torch.matmul(self.omega, z) + self.b
        x = self.alpha * np.sqrt(2.0 / self.D) * torch.cos(linear_combination)
        x = self.process(x)
        return self.theta_base + x


class MLPSoftSharing(ParameterSharing):
    """
    Parameter sharing using a Multi-Layer Perceptron (MLP) decoder.
    
    This class implements a more flexible parameter sharing approach by using
    a neural network to map from the latent space to the full parameter space.
    The MLP can learn complex non-linear mappings that may be more expressive
    than simple linear projections.
    
    The decoder network consists of:
    - Input layer: maps from latent dimension d to first hidden layer
    - Hidden layers: configurable number and sizes with optional activation
    - Output layer: maps to full parameter dimension D (no activation)
    
    Input types:
        - Model: torch.nn.Module
        - d: int (number of shared parameters)
        - Scaling factor alpha: float
        - Hidden dimensions: List[int] (number of hidden dimensions)
        - Use activation: bool (whether to use activation)
        - Activation: str (activation function)
    
    Output types:
        - Full parameter vector: torch.Tensor of shape (D,) where D is the
          total number of model parameters
    """
    
    def __init__(self, model: torch.nn.Module, d: int, 
                 hidden_dims: List[int] = [32, 16], use_activation: bool = True, alpha: float = 1.0,
                 activation: str = 'relu', device: str = 'cuda', seed: int = 0) -> None:
        super().__init__(model, d, alpha, device, seed)

        self.hidden_dims = hidden_dims
        self.activation = activation
        self.use_activation = use_activation
        if use_activation:
            if activation == 'relu':
                self.activation_function = nn.ReLU()
            elif activation == 'tanh':
                self.activation_function = nn.Tanh()
            elif activation == 'gelu':
                self.activation_function = nn.GELU()
            elif activation == 'leaky_relu':
                self.activation_function = nn.LeakyReLU()
            else:
                raise ValueError(f"Activation function {activation} not supported")
        else:
            self.activation_function = None
        self.decoder = None
        self.init()

    def init(self) -> None:
        """
        Initialize decoder weights using He normal initialization.
        
        Each layer is initialized with standard deviation sqrt(1/fan_in)
        to maintain reasonable activation magnitudes.
        """
        layers = []
        # First layer: input to first hidden
        layer = nn.Linear(self.d, self.hidden_dims[0], bias=False)
        # Normalize weights to unit norm
        with torch.no_grad():
            layer.weight.data = layer.weight.data / torch.norm(layer.weight.data, dim=1, keepdim=True)
            layer.weight.data = layer.weight.data / self.d ** 0.5
        layers.append(layer)
        if self.use_activation:
            layers.append(self.activation_function)
        
        # Intermediate hidden layers
        for i in range(len(self.hidden_dims) - 1):
            layer = nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1], bias=False)
            # Normalize weights to unit norm
            with torch.no_grad():
                layer.weight.data = layer.weight.data / torch.norm(layer.weight.data, dim=1, keepdim=True)
                layer.weight.data = layer.weight.data / self.hidden_dims[i] ** 0.5
            layers.append(layer)
            if self.use_activation:
                layers.append(self.activation_function)
        
        # Final layer: last hidden to output (no activation, as parameters can be any real values)
        layer = nn.Linear(self.hidden_dims[-1], self.D, bias=False)
        # Normalize weights to unit norm
        with torch.no_grad():
            layer.weight.data = layer.weight.data / torch.norm(layer.weight.data, dim=1, keepdim=True)
            layer.weight.data = layer.weight.data / self.hidden_dims[-1] ** 0.5
        layers.append(layer)
        
        self.decoder = nn.Sequential(*layers)
        self.decoder.to(self.device)

    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using MLP decoder.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,) computed as theta_base + MLP(z)
        """
        z = self._to_tensor(z)
        with torch.no_grad():
            x = self.decoder(z)
        x = self.process(x)
        return self.theta_base + x
    
class HyperNetworkSoftSharing(ParameterSharing):
    """
    Parameter sharing using HyperNetworks with layer-specific embeddings.
    
    This class implements a sophisticated parameter sharing approach where a shared
    neural network generates parameters for different layers of the target model.
    Each layer is associated with a unique embedding that captures its identity,
    and the hypernetwork takes both the latent vector and layer embedding as input.
    
    The approach uses:
    - Layer embeddings: unique identifiers for each model layer (sinusoidal or random)
    - Shared MLP decoder: generates parameters for all layers conditioned on embeddings
    - Efficient batching: processes all layers simultaneously
    
    Input types:
        - Model: torch.nn.Module
        - d: int (number of shared parameters)
        - Scaling factor alpha: float
        - Hidden dimensions: List[int] (number of hidden dimensions)
        - Embedding initialization: str (embedding initialization)
    
    Output types:
        - Full parameter vector: torch.Tensor of shape (D,) where D is the
          total number of model parameters across all layers
    """
    
    def __init__(self, model: torch.nn.Module, d: int, 
                 hidden_dims: List[int] = [32, 16], emb_init: str = 'sinusoidal', alpha: float = 1.0,
                 device: str = 'cuda', seed: int = 0) -> None:
        super().__init__(model, d, alpha, device, seed)

        self.hidden_dims = hidden_dims
        
        layer_sizes = []
        for layer in model.parameters():
            layer_sizes.append(layer.numel())

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.embed_dim = 16

        if emb_init == "random":
            # Option 1: Random initialization with seed
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            embeddings = torch.randn(self.num_layers, self.embed_dim, generator=generator, device=device) * 0.1
            self.register_parameter('embeddings', nn.Parameter(embeddings))
        elif emb_init == "sinusoidal":
            # Precompute sinusoidal embeddings for all layers
            positions = torch.arange(self.num_layers, dtype=torch.float32) / (self.num_layers - 1)  # Normalized [0,1]
            freqs = torch.arange(self.embed_dim // 2, dtype=torch.float32)  # Half for sin, half for cos
            freqs = 2.0 ** freqs * torch.pi  # Exponentially increasing frequencies
            pos = positions.unsqueeze(1)  # Shape: (num_layers, 1)
            freq = freqs.unsqueeze(0)  # Shape: (1, embed_dim//2)
            sin = torch.sin(pos * freq)  # Shape: (num_layers, embed_dim//2)
            cos = torch.cos(pos * freq)  # Shape: (num_layers, embed_dim//2)
            embeddings = torch.cat([sin, cos], dim=1)  # Shape: (num_layers, embed_dim)
            self.register_buffer('embeddings', embeddings)
        else:
            raise ValueError("init_type must be 'random' or 'sinusoidal'")

        # Shared MLP: input = d + embed_dim, hidden layers, output = max(layer_sizes)
        max_output_size = max(layer_sizes)
        self.max_output_size = max_output_size
        self.decoder = nn.Sequential(
            nn.Linear(self.d + self.embed_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], max_output_size)
        )
        print("Decoder parameters number:", sum(p.numel() for p in self.decoder.parameters()))

    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using layer-specific hypernetwork.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,) computed as:
            theta_base + concatenation of layer-specific parameters generated by
            the hypernetwork conditioned on layer embeddings
        """
        z = self._to_tensor(z).unsqueeze(0)
        batch_size = z.size(0)
        
        # Expand z to [batch_size, num_layers, d]
        z_expanded = z.unsqueeze(1).expand(-1, self.num_layers, -1)
        
        # Expand layer embeddings to [batch_size, num_layers, embed_dim]
        embeds = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate: [batch_size, num_layers, d + embed_dim]
        mlp_input = torch.cat([z_expanded, embeds], dim=-1)
        
        # Flatten for MLP: [batch_size * num_layers, d + embed_dim]
        mlp_input = mlp_input.view(-1, mlp_input.size(-1))
        
        # Single MLP call: [batch_size * num_layers, max_output_size]
        params = self.decoder(mlp_input)
        
        # Reshape and extract layer-specific parameters: [batch_size, layer_i_size]
        params = params.view(batch_size, self.num_layers, self.max_output_size)
        outputs = [params[:, i, :self.layer_sizes[i]] for i in range(self.num_layers)]

        x = torch.cat(outputs, dim=1).squeeze(0)
        x = self.process(x)
        return self.theta_base + x


class SparseProjection(ParameterSharing):
    """Sparse random projection: P is D x d sparse matrix with density 'sparsity'.
    Non-zeros ~ N(0,1), then columns normalized to unit length.
    Input: theta_d (batch, d) or (d,), Output: delta_theta_D (batch, D) or (D,).
    Full params: theta_0 + output.
    """
    def __init__(self, model: torch.nn.Module, d: int, 
                 sparsity: float = 0.1, alpha: float = 1.0, device: str = 'cuda', seed: int = 0):
        super().__init__(model, d, alpha, device, seed)
        self.sparsity = sparsity
        
        # Generate indices for non-zeros: random positions with seed
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        num_nz = int(self.D * self.d * self.sparsity)
        row_idx = torch.randint(0, self.D, (num_nz,), device=device, generator=generator)
        col_idx = torch.randint(0, self.d, (num_nz,), device=device, generator=generator)
        
        # Values: N(0,1)
        values = torch.randn(num_nz, device=device, generator=generator)
        
        # Sparse tensor in COO
        P_coo = torch.sparse_coo_tensor(
            torch.stack([row_idx, col_idx]), values,
            size=(self.D, self.d), device=device
        )
        
        # Convert to dense temporarily for column normalization
        P_dense = P_coo.to_dense()
        # Normalize columns
        P_dense = P_dense / torch.norm(P_dense, dim=0, keepdim=True)
        # Back to sparse (but sparsity may change slightly)
        self.register_buffer('P', P_dense.to_sparse())
    
    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using sparse projection.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,) computed as theta_base + P @ z
        """
        z = self._to_tensor(z)
        # Convert to dense for matmul (tradeoff)
        P_dense = self.P.to_dense()
        x = P_dense @ z
        x = self.process(x)
        return self.theta_base + x


class FWHT(Function):
    """Fast Walsh-Hadamard Transform (FWHT) as autograd Function.
    Assumes input length is power of 2. Normalizes by sqrt(length).
    """
    @staticmethod
    def forward(ctx, input):
        # Convert to numpy for faster computation (20x speedup vs pure torch)
        length = input.shape[-1]
        assert np.log2(length).is_integer(), "Length must be power of 2"
        result = input.detach().numpy()
        
        bit = length
        for _ in range(int(np.log2(length))):
            bit //= 2
            for i in range(length):
                if i & bit == 0:
                    j = i | bit
                    temp = result[i]
                    result[i] += result[j]
                    result[j] = temp - result[j]
        
        result /= np.sqrt(length)
        ctx.save_for_backward(torch.from_numpy(result).to(input.device))  # Not strictly needed since self-inverse
        return torch.from_numpy(result).to(input.device)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Self-inverse: apply same transform
        return FWHT.forward(ctx, grad_output)


class FastfoodProjection(ParameterSharing):
    """Fastfood random projection: Approximates G x where G ~ N(0, I_d).
    Input: theta_d (d,), Output: delta_theta_D (D,). D must be power of 2; pad if needed.
    Full params: theta_0 + output.
    """
    def __init__(self, model: torch.nn.Module, d: int, 
                 alpha: float = 1.0, device: str = 'cuda', seed: int = 0):
        super().__init__(model, d, alpha, device, seed)
        assert np.log2(self.D).is_integer(), "Full parameter dimension D must be power of 2"
        self.fwht = FWHT.apply
        
        # Generate fixed random factors (frozen) with seed
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        
        # Permutation Pi
        self.register_buffer('perm', torch.randperm(self.D, device=device, generator=generator))
        
        # Diagonal B: ±1
        self.register_buffer('B', torch.sign(torch.rand(self.D, device=device, generator=generator) - 0.5))
        
        # Diagonal H: abs ~ N(0,1), random signs (for d slice)
        h_abs = torch.abs(torch.randn(self.d, device=device, generator=generator))
        h_sign = torch.sign(torch.rand(self.d, device=device, generator=generator) - 0.5)
        self.register_buffer('H', h_abs * h_sign)
        
        # Scaling factor
        self.scale = 1.0 / np.sqrt(self.d)
    
    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using Fastfood projection.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,) computed as theta_base + Fastfood(z)
        """
        z = self._to_tensor(z)
        
        # Pad z to D dimensions with zeros
        z_padded = torch.zeros(self.D, device=self.device)
        z_padded[:self.d] = z * self.H  # Apply H scaling first
        
        # Step 1: FWHT on padded input
        x_fwht = self.fwht(z_padded.unsqueeze(0)).squeeze(0)  # (D,)
        
        # Step 2: Permute
        x_perm = x_fwht[self.perm]
        
        # Step 3: Scale by B (elementwise multiply)
        x_b = x_perm * self.B
        
        # Step 4: Inverse FWHT (same as forward)
        x_inv_fwht = self.fwht(x_b.unsqueeze(0)).squeeze(0)
        
        # Step 5: Apply scaling
        x = x_inv_fwht * self.scale
        x = self.process(x)
        return self.theta_base + x


class HardWeightSharingCodebook(ParameterSharing):
    """
    Hard weight sharing using a learnable codebook.
    
    This class implements hard parameter sharing where parameters are quantized to
    discrete codes from a learnable codebook. Each parameter position is assigned
    to one of K codebook entries, and the full parameter vector is reconstructed
    by looking up these assignments in the codebook.
    
    The mapping is: theta[i] = codebook[assignments[i]] for all i
    
    Args:
        model: PyTorch model
        d: Number of codebook entries (latent dimension)
        seed: Random seed for reproducible assignment generation
        alpha: Scaling factor for parameters
        device: Device for computations
        assignment_strategy: How to initialize assignments ('random', 'kmeans', 'uniform')
    """
    
    def __init__(self, model: torch.nn.Module, d: int, seed: int = 0,
                 alpha: float = 1.0, device: str = 'cuda', 
                 assignment_strategy: str = 'random') -> None:
        super().__init__(model, d, alpha, device, seed)
        self.assignment_strategy = assignment_strategy
        
        self._init_assignments()
        
        print(self.get_codebook_usage())
    
    def _init_assignments(self) -> None:
        """Initialize parameter assignments to codebook entries."""
        if self.assignment_strategy == 'random':
            # Random assignment with seed
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
            assignments = torch.randint(0, self.d, (self.D,), device=self.device, generator=generator)
        elif self.assignment_strategy == 'uniform':
            # Uniform distribution across codebook
            assignments = torch.arange(self.D, device=self.device) % self.d
        elif self.assignment_strategy == 'kmeans':
            # Use k-means clustering on current parameters
            try:
                from sklearn.cluster import KMeans
                params_np = self.theta_base.detach().cpu().numpy().reshape(-1, 1)
                kmeans = KMeans(n_clusters=self.d, random_state=self.seed, n_init=10)
                assignments = kmeans.fit_predict(params_np)
                assignments = torch.from_numpy(assignments).to(self.device)
                
                # Initialize codebook with cluster centers
                centers = torch.from_numpy(kmeans.cluster_centers_.flatten()).float()
                with torch.no_grad():
                    self.codebook.data.copy_(centers.to(self.device))
            except ImportError:
                print("sklearn not available, falling back to random assignment")
                generator = torch.Generator(device=self.device)
                generator.manual_seed(self.seed)
                assignments = torch.randint(0, self.d, (self.D,), device=self.device, generator=generator)
        else:
            raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy}")
        
        # Store assignments as buffer (non-learnable)
        self.register_buffer('assignments', assignments)
    
    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector (codebook) to full parameter space using hard assignment.
        
        Args:
            z: Codebook vector of shape (d,) - these become the codebook entries
            
        Returns:
            Full parameter tensor of shape (D,) where each position gets the value
            from the corresponding codebook entry: theta[i] = z[assignments[i]]
        """
        z = self._to_tensor(z)
        
        # Use the input z as the codebook values and look up based on assignments
        x = z[self.assignments]
        x = self.process(x)
        return self.theta_base + x
    
    def get_codebook_usage(self) -> torch.Tensor:
        """Return how many parameters are assigned to each codebook entry."""
        usage = torch.zeros(self.d, device=self.device)
        for i in range(self.d):
            usage[i] = (self.assignments == i).sum()
        return usage
    
    def get_compression_ratio(self) -> float:
        """Return the compression ratio achieved by the codebook."""
        return self.D / self.d


class BlockwiseDenseProjection(ParameterSharing):
    """
    Blockwise dense projection: Divides parameter space into blocks and applies
    separate dense projections to each block.
    
    This class implements structured parameter sharing by partitioning the full
    parameter vector into B blocks of (approximately) equal size, then applying
    independent dense random projections to each block. This allows for more
    flexible parameter sharing while maintaining some structure.
    
    The mapping is: theta = theta_base + concat([P_i @ z_i for i in range(B)])
    where P_i are block-specific projection matrices and z_i are block-specific
    latent vectors.
    
    Args:
        model: PyTorch model
        d: Total latent dimension (will be divided among blocks)
        num_blocks: Number of blocks to divide parameters into
        alpha: Scaling factor for parameters
        normalize: Whether to normalize projection matrices
        device: Device for computations
        block_strategy: How to divide parameters ('equal', 'layer_wise', 'random')
    """
    
    def __init__(self, model: torch.nn.Module, d: int,
                 num_blocks: int = 4, alpha: float = 1.0, normalize: bool = False,
                 device: str = 'cuda', block_strategy: str = 'equal', seed: int = 0) -> None:
        super().__init__(model, d, alpha, device, seed)
        self.num_blocks = num_blocks
        self.normalize = normalize
        self.block_strategy = block_strategy
        
        # Divide latent dimension among blocks
        self.d_per_block = d // num_blocks
        self.d_remainder = d % num_blocks
        
        # Calculate block dimensions and create block boundaries
        self._create_blocks()
        
        # Initialize projection matrices for each block
        self.projection_matrices = nn.ParameterList()
        self._init_projections()
    
    def _create_blocks(self) -> None:
        """Create parameter blocks based on the specified strategy."""
        if self.block_strategy == 'equal':
            # Divide parameters into equal-sized blocks
            block_size = self.D // self.num_blocks
            self.block_boundaries = []
            start = 0
            for i in range(self.num_blocks):
                end = start + block_size
                if i == self.num_blocks - 1:  # Last block gets remainder
                    end = self.D
                self.block_boundaries.append((start, end))
                start = end
                
        elif self.block_strategy == 'layer_wise':
            # Try to align blocks with model layers
            self.block_boundaries = self._get_layer_boundaries()
            
        elif self.block_strategy == 'random':
            # Random permutation then equal division with seed
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
            perm = torch.randperm(self.D, device=self.device, generator=generator)
            self.register_buffer('param_permutation', perm)
            
            block_size = self.D // self.num_blocks
            self.block_boundaries = []
            start = 0
            for i in range(self.num_blocks):
                end = start + block_size
                if i == self.num_blocks - 1:
                    end = self.D
                self.block_boundaries.append((start, end))
                start = end
        else:
            raise ValueError(f"Unknown block strategy: {self.block_strategy}")
    
    def _get_layer_boundaries(self) -> list:
        """Get block boundaries aligned with model layers."""
        boundaries = []
        start = 0
        params_per_layer = []
        
        # Calculate parameters per layer
        for param in self.model.parameters():
            layer_size = param.numel()
            params_per_layer.append(layer_size)
        
        # Group layers into blocks
        total_layers = len(params_per_layer)
        layers_per_block = max(1, total_layers // self.num_blocks)
        
        current_block_start = 0
        for block_idx in range(self.num_blocks):
            start_layer = block_idx * layers_per_block
            if block_idx == self.num_blocks - 1:
                # Last block gets all remaining layers
                end_layer = total_layers
            else:
                end_layer = min((block_idx + 1) * layers_per_block, total_layers)
            
            # Calculate parameter range for this block
            block_start = current_block_start
            block_end = current_block_start + sum(params_per_layer[start_layer:end_layer])
            
            boundaries.append((block_start, block_end))
            current_block_start = block_end
            
            if block_end >= self.D:
                break
        
        return boundaries
    
    def _init_projections(self) -> None:
        """Initialize projection matrices for each block with seed."""
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        
        for i, (start, end) in enumerate(self.block_boundaries):
            block_size = end - start
            
            # Determine latent dimension for this block
            if i < self.d_remainder:
                block_d = self.d_per_block + 1
            else:
                block_d = self.d_per_block
            
            # Create projection matrix: block_size x block_d
            P = torch.randn(block_size, block_d, device=self.device, generator=generator)
            
            if self.normalize:
                P = P / P.norm(dim=0, keepdim=True)
                P = P / (block_d ** 0.5)
            
            self.projection_matrices.append(nn.Parameter(P))
    
    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using blockwise projections.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,) computed as:
            theta_base + concat([P_i @ z_i for each block i])
        """
        z = self._to_tensor(z)
        
        # Split latent vector into block-specific parts
        z_blocks = self._split_latent_vector(z)
        
        # Apply projection for each block
        block_outputs = []
        for i, (z_block, P) in enumerate(zip(z_blocks, self.projection_matrices)):
            block_output = P @ z_block
            block_outputs.append(block_output)
        
        # Concatenate block outputs
        x = torch.cat(block_outputs, dim=0)
        
        # Apply permutation if using random strategy
        if self.block_strategy == 'random':
            x_permuted = torch.zeros_like(x)
            x_permuted[self.param_permutation] = x
            x = x_permuted
        
        x = self.process(x)
        return self.theta_base + x
    
    def _split_latent_vector(self, z: torch.Tensor) -> List[torch.Tensor]:
        """Split latent vector into block-specific parts."""
        z_blocks = []
        start_idx = 0
        
        for i in range(self.num_blocks):
            # Determine block latent dimension
            if i < self.d_remainder:
                block_d = self.d_per_block + 1
            else:
                block_d = self.d_per_block
            
            end_idx = start_idx + block_d
            z_block = z[start_idx:end_idx]
            z_blocks.append(z_block)
            start_idx = end_idx
        
        return z_blocks
    
    def get_block_info(self) -> dict:
        """Return information about the block structure."""
        info = {
            'num_blocks': self.num_blocks,
            'block_strategy': self.block_strategy,
            'total_latent_dim': self.d,
            'latent_per_block': self.d_per_block,
            'remainder_latent': self.d_remainder,
            'blocks': []
        }
        
        for i, (start, end) in enumerate(self.block_boundaries):
            block_d = self.d_per_block + (1 if i < self.d_remainder else 0)
            block_info = {
                'block_id': i,
                'param_range': (start, end),
                'param_size': end - start,
                'latent_dim': block_d,
                'compression_ratio': (end - start) / block_d
            }
            info['blocks'].append(block_info)
        
        return info
    
    def get_total_compression_ratio(self) -> float:
        """Return the overall compression ratio."""
        return self.D / self.d


class LearnedBasisProjection(ParameterSharing):
    """
    Learned Basis Projections: Learn optimal parameter directions for projection.
    
    This class learns an optimal basis for parameter projection by analyzing
    parameter evolution during training. Instead of using random projections,
    it identifies the most important directions in parameter space using:
    - PCA on parameter history
    - Gradient-based importance
    - Fisher Information Matrix eigenvectors
    - Parameter variance analysis
    
    The learned basis can achieve much higher compression ratios than random
    projections by focusing on directions that actually matter for the model.
    
    Args:
        model: PyTorch model
        d: Number of basis vectors (latent dimension)
        basis_method: How to learn basis ('pca', 'gradient_pca', 'fisher_info', 'variance')
        update_freq: How often to update the basis (in forward calls)
        history_length: How many parameter snapshots to keep for analysis
        alpha: Scaling factor
        device: Device for computations
    """
    
    def __init__(self, model: torch.nn.Module, d: int,
                 basis_method: str = 'pca', update_freq: int = 100, 
                 history_length: int = 50, alpha: float = 1.0, device: str = 'cuda', seed: int = 0) -> None:
        super().__init__(model, d, alpha, device, seed)
        self.basis_method = basis_method
        self.update_freq = update_freq
        self.history_length = history_length
        self.forward_count = 0
        
        # Parameter history for basis learning
        self.param_history = []
        self.gradient_history = []
        
        # Learned basis (initially random)
        self.basis_matrix = None
        self.basis_learned = False
        self._init_basis()
        
        # Statistics for basis updates
        self.last_update_step = 0
        self.basis_update_count = 0
    
    def _init_basis(self) -> None:
        """Initialize basis with random projection using seed."""
        # Start with random basis, will be updated during training
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        basis = torch.randn(self.D, self.d, device=self.device, generator=generator)
        basis = basis / basis.norm(dim=0, keepdim=True)  # Normalize columns
        self.register_buffer('basis_matrix', basis)
    
    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using learned basis.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,) computed as:
            theta_base + learned_basis @ z
        """
        z = self._to_tensor(z)
        
        # Apply learned basis projection
        x = self.basis_matrix @ z
        x = self.process(x)
        
        # Update basis periodically
        self.forward_count += 1
        if self.forward_count % self.update_freq == 0:
            self._maybe_update_basis()
        
        return self.theta_base + x
    
    def update_parameter_history(self, current_params: torch.Tensor = None) -> None:
        """Update parameter history for basis learning."""
        if current_params is None:
            current_params = params_to_vector(self.model.parameters())
        
        # Add to history
        self.param_history.append(current_params.detach().clone())
        
        # Maintain history length
        if len(self.param_history) > self.history_length:
            self.param_history.pop(0)
    
    def update_gradient_history(self) -> None:
        """Update gradient history for gradient-based methods."""
        # Collect current gradients
        grad_vector = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_vector.append(param.grad.view(-1))
            else:
                grad_vector.append(torch.zeros_like(param.view(-1)))
        
        current_grad = torch.cat(grad_vector)
        self.gradient_history.append(current_grad.detach().clone())
        
        # Maintain history length
        if len(self.gradient_history) > self.history_length:
            self.gradient_history.pop(0)
    
    def _maybe_update_basis(self) -> None:
        """Update basis if enough data is available."""
        if self.basis_method in ['pca', 'variance'] and len(self.param_history) >= 10:
            self._update_basis_pca()
        elif self.basis_method in ['gradient_pca'] and len(self.gradient_history) >= 10:
            self._update_basis_gradient()
        elif self.basis_method == 'fisher_info' and len(self.gradient_history) >= 10:
            self._update_basis_fisher()
    
    def _update_basis_pca(self) -> None:
        """Update basis using PCA on parameter variations."""
        if len(self.param_history) < 2:
            return
        
        try:
            # Stack parameter history: (history_length, D)
            param_matrix = torch.stack(self.param_history)
            
            if self.basis_method == 'pca':
                # Center the data
                param_centered = param_matrix - param_matrix.mean(dim=0, keepdim=True)
            else:  # variance method
                # Use parameter variance directly
                param_centered = param_matrix
            
            # Compute covariance matrix: (D, D)
            if param_centered.shape[0] > 1:
                cov_matrix = torch.matmul(param_centered.t(), param_centered) / (param_centered.shape[0] - 1)
            else:
                return
            
            # Eigendecomposition (get top-d eigenvectors)
            eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalue magnitude (descending)
            sorted_indices = torch.argsort(eigenvals.abs(), descending=True)
            top_eigenvecs = eigenvecs[:, sorted_indices[:self.d]]
            top_eigenvals = eigenvals[sorted_indices[:self.d]]
            
            # Update basis
            self.basis_matrix.data.copy_(top_eigenvecs)
            self.basis_learned = True
            self.basis_update_count += 1
            
            # Print update info
            explained_variance = top_eigenvals.sum() / eigenvals.sum()
            print(f"Basis updated (PCA): {explained_variance:.3f} variance explained with {self.d} components")
            
        except Exception as e:
            print(f"PCA basis update failed: {e}, keeping previous basis")
    
    def _update_basis_gradient(self) -> None:
        """Update basis using PCA on gradients."""
        if len(self.gradient_history) < 2:
            return
        
        try:
            # Stack gradient history: (history_length, D)
            grad_matrix = torch.stack(self.gradient_history)
            
            # Center the gradients
            grad_centered = grad_matrix - grad_matrix.mean(dim=0, keepdim=True)
            
            # Compute covariance matrix
            if grad_centered.shape[0] > 1:
                cov_matrix = torch.matmul(grad_centered.t(), grad_centered) / (grad_centered.shape[0] - 1)
            else:
                return
            
            # Eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalue magnitude (descending)
            sorted_indices = torch.argsort(eigenvals.abs(), descending=True)
            top_eigenvecs = eigenvecs[:, sorted_indices[:self.d]]
            top_eigenvals = eigenvals[sorted_indices[:self.d]]
            
            # Update basis
            self.basis_matrix.data.copy_(top_eigenvecs)
            self.basis_learned = True
            self.basis_update_count += 1
            
            # Print update info
            explained_variance = top_eigenvals.sum() / eigenvals.sum()
            print(f"Basis updated (Gradient PCA): {explained_variance:.3f} variance explained")
            
        except Exception as e:
            print(f"Gradient PCA basis update failed: {e}")
    
    def _update_basis_fisher(self) -> None:
        """Update basis using Fisher Information Matrix approximation."""
        if len(self.gradient_history) < 2:
            return
        
        try:
            # Approximate Fisher Information Matrix using gradient outer products
            fisher_matrix = torch.zeros(self.D, self.D, device=self.device)
            
            for grad in self.gradient_history[-10:]:  # Use recent gradients
                fisher_matrix += torch.outer(grad, grad) / len(self.gradient_history[-10:])
            
            # Eigendecomposition of Fisher Information Matrix
            eigenvals, eigenvecs = torch.linalg.eigh(fisher_matrix)
            
            # Sort by eigenvalue (descending) - high Fisher info = important directions
            sorted_indices = torch.argsort(eigenvals, descending=True)
            top_eigenvecs = eigenvecs[:, sorted_indices[:self.d]]
            top_eigenvals = eigenvals[sorted_indices[:self.d]]
            
            # Update basis
            self.basis_matrix.data.copy_(top_eigenvecs)
            self.basis_learned = True
            self.basis_update_count += 1
            
            # Print update info
            total_fisher = eigenvals.sum()
            captured_fisher = top_eigenvals.sum()
            print(f"Basis updated (Fisher): {captured_fisher/total_fisher:.3f} Fisher info captured")
            
        except Exception as e:
            print(f"Fisher basis update failed: {e}")
    
    def get_basis_quality_metrics(self) -> dict:
        """Return metrics about the learned basis quality."""
        metrics = {
            'basis_learned': self.basis_learned,
            'basis_updates': self.basis_update_count,
            'param_history_length': len(self.param_history),
            'gradient_history_length': len(self.gradient_history),
            'basis_method': self.basis_method
        }
        
        # Compute basis orthogonality
        if self.basis_matrix is not None:
            gram_matrix = self.basis_matrix.t() @ self.basis_matrix
            identity = torch.eye(self.d, device=self.device)
            orthogonality_error = (gram_matrix - identity).norm().item()
            metrics['orthogonality_error'] = orthogonality_error
            
            # Compute condition number
            singular_values = torch.linalg.svdvals(self.basis_matrix)
            condition_number = (singular_values.max() / singular_values.min()).item()
            metrics['condition_number'] = condition_number
        
        return metrics
    
    def force_basis_update(self) -> None:
        """Force an immediate basis update."""
        self._maybe_update_basis()
    
    def reset_basis(self) -> None:
        """Reset to random basis and clear history."""
        self._init_basis()
        self.param_history.clear()
        self.gradient_history.clear()
        self.basis_learned = False
        self.basis_update_count = 0
        print("Basis reset to random initialization")
    
    def save_basis(self, filepath: str) -> None:
        """Save the learned basis to file."""
        torch.save({
            'basis_matrix': self.basis_matrix,
            'basis_method': self.basis_method,
            'basis_learned': self.basis_learned,
            'basis_updates': self.basis_update_count
        }, filepath)
    
    def load_basis(self, filepath: str) -> None:
        """Load a previously saved basis."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.basis_matrix.data.copy_(checkpoint['basis_matrix'])
        self.basis_learned = checkpoint['basis_learned']
        self.basis_update_count = checkpoint.get('basis_updates', 0)
        print(f"Loaded basis with {self.basis_update_count} previous updates")


class ParameterizedBasisProjection(ParameterSharing):
    """
    Parameterized Basis Projection: The basis matrix itself is part of the latent space.
    
    This class treats the projection basis as learnable parameters that are optimized
    together with the latent coordinates using non-differentiable methods like ES.
    
    The latent vector z is split into two parts:
    1. Basis parameters (z_basis): Define the projection matrix
    2. Coordinate parameters (z_coords): Define the coordinates in that basis
    
    Latent space structure:
    z = [z_basis, z_coords] where:
    - z_basis: (D * k) parameters defining k basis vectors
    - z_coords: (k) parameters defining coordinates in the basis
    
    The mapping is: theta = theta_base + reshape(z_basis, (D, k)) @ z_coords
    
    This allows ES to jointly optimize both:
    - What the basis vectors should be (z_basis)
    - How much of each basis vector to use (z_coords)
    
    Args:
        model: PyTorch model
        k: Number of basis vectors
        basis_parameterization: How to parameterize basis ('direct', 'orthogonal', 'low_rank')
        basis_constraint: Constraint on basis vectors ('unit_norm', 'orthogonal', 'none')
        alpha: Scaling factor
        device: Device for computations
    """
    
    def __init__(self, model: torch.nn.Module, k: int,
                 basis_parameterization: str = 'direct', basis_constraint: str = 'unit_norm',
                 alpha: float = 1.0, device: str = 'cuda', seed: int = 0) -> None:
        
        # Get actual parameter count
        temp_D = sum(p.numel() for p in model.parameters())
        
        # Total latent dimension = basis parameters + coordinates
        if basis_parameterization == 'direct':
            d_basis = temp_D * k  # D * k for full basis
        elif basis_parameterization == 'orthogonal':
            d_basis = k * (k + 1) // 2  # Lower triangular for QR decomposition
        elif basis_parameterization == 'low_rank':
            d_basis = 2 * k * int(np.sqrt(temp_D))  # U @ V.T factorization
        else:
            d_basis = temp_D * k
            
        d_total = d_basis + k  # basis params + coordinates
        
        super().__init__(model, d_total, alpha, device, seed)
        
        self.k = k  # Number of basis vectors
        self.d_basis = d_basis  # Dimension for basis parameterization
        self.d_coords = k  # Dimension for coordinates
        self.basis_parameterization = basis_parameterization
        self.basis_constraint = basis_constraint
        
        # Initialize reference basis for orthogonal parameterization
        if basis_parameterization == 'orthogonal':
            self._init_reference_basis()
    
    def _init_reference_basis(self) -> None:
        """Initialize reference basis for orthogonal parameterization with seed."""
        # Create initial orthogonal basis using QR decomposition
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        random_matrix = torch.randn(self.D, self.k, device=self.device, generator=generator)
        Q, _ = torch.linalg.qr(random_matrix)
        self.register_buffer('reference_basis', Q)
    
    def _parameterize_basis(self, z_basis: torch.Tensor) -> torch.Tensor:
        """Convert basis parameters to actual basis matrix."""
        
        if self.basis_parameterization == 'direct':
            # Direct parameterization: reshape to (D, k)
            basis_matrix = z_basis.view(self.D, self.k)
            
        elif self.basis_parameterization == 'orthogonal':
            # Orthogonal parameterization using Givens rotations or QR
            basis_matrix = self._orthogonal_basis_from_params(z_basis)
            
        elif self.basis_parameterization == 'low_rank':
            # Low-rank factorization: U @ V.T
            basis_matrix = self._low_rank_basis_from_params(z_basis)
            
        else:
            raise ValueError(f"Unknown basis parameterization: {self.basis_parameterization}")
        
        # Apply constraints
        basis_matrix = self._apply_basis_constraints(basis_matrix)
        
        return basis_matrix
    
    def _orthogonal_basis_from_params(self, z_basis: torch.Tensor) -> torch.Tensor:
        """Create orthogonal basis using parameterized rotations."""
        # Use Givens rotations to create orthogonal matrix
        # This is a simplified version - could use more sophisticated parameterizations
        
        # Start with reference basis
        basis = self.reference_basis.clone()
        
        # Apply parameterized rotations
        param_idx = 0
        for i in range(self.k):
            for j in range(i + 1, self.k):
                if param_idx < len(z_basis):
                    # Givens rotation angle
                    theta = z_basis[param_idx] * np.pi  # Scale to full rotation
                    
                    # Apply Givens rotation to columns i and j
                    cos_theta = torch.cos(theta)
                    sin_theta = torch.sin(theta)
                    
                    # Rotation matrix
                    col_i = basis[:, i].clone()
                    col_j = basis[:, j].clone()
                    
                    basis[:, i] = cos_theta * col_i - sin_theta * col_j
                    basis[:, j] = sin_theta * col_i + cos_theta * col_j
                    
                    param_idx += 1
        
        return basis
    
    def _low_rank_basis_from_params(self, z_basis: torch.Tensor) -> torch.Tensor:
        """Create basis using low-rank factorization U @ V.T."""
        # Split parameters into U and V factors
        sqrt_D = int(np.sqrt(self.D))
        u_params = z_basis[:self.k * sqrt_D].view(sqrt_D, self.k)
        v_params = z_basis[self.k * sqrt_D:].view(sqrt_D, self.k)
        
        # Create basis via outer product and reshape
        basis_matrix = torch.zeros(self.D, self.k, device=self.device)
        
        for i in range(self.k):
            # Outer product of u[:, i] and v[:, i], then flatten
            outer = torch.outer(u_params[:, i], v_params[:, i])
            if outer.numel() <= self.D:
                basis_matrix[:outer.numel(), i] = outer.flatten()
            else:
                basis_matrix[:, i] = outer.flatten()[:self.D]
        
        return basis_matrix
    
    def _apply_basis_constraints(self, basis_matrix: torch.Tensor) -> torch.Tensor:
        """Apply constraints to the basis matrix."""
        
        if self.basis_constraint == 'unit_norm':
            # Normalize each column to unit norm
            basis_matrix = basis_matrix / (basis_matrix.norm(dim=0, keepdim=True) + 1e-8)
            
        elif self.basis_constraint == 'orthogonal':
            # Orthogonalize using QR decomposition
            Q, _ = torch.linalg.qr(basis_matrix)
            basis_matrix = Q
            
        elif self.basis_constraint == 'none':
            # No constraints
            pass
        
        else:
            raise ValueError(f"Unknown basis constraint: {self.basis_constraint}")
        
        return basis_matrix
    
    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using parameterized basis.
        
        Args:
            z: Latent vector of shape (d_basis + k,) = (d_total,)
               First d_basis elements define the basis
               Last k elements define the coordinates
            
        Returns:
            Full parameter tensor of shape (D,) computed as:
            theta_base + basis_matrix @ coordinates
        """
        z = self._to_tensor(z)
        
        # Split latent vector into basis and coordinate parts
        z_basis = z[:self.d_basis]
        z_coords = z[self.d_basis:]
        
        # Parameterize the basis matrix from z_basis
        basis_matrix = self._parameterize_basis(z_basis)
        
        # Apply the basis to coordinates
        x = basis_matrix @ z_coords
        x = self.process(x)
        
        return self.theta_base + x
    
    def get_basis_from_latent(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract the basis matrix from a latent vector."""
        z = self._to_tensor(z)
        z_basis = z[:self.d_basis]
        return self._parameterize_basis(z_basis)
    
    def get_coordinates_from_latent(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract the coordinates from a latent vector."""
        z = self._to_tensor(z)
        return z[self.d_basis:]
    
    def create_latent_from_basis_coords(self, basis_matrix: torch.Tensor, 
                                      coordinates: torch.Tensor) -> torch.Tensor:
        """Create a latent vector from a basis matrix and coordinates."""
        # This is more complex - would need to invert the parameterization
        # For now, return random basis parameters (this is a limitation)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        z_basis = torch.randn(self.d_basis, device=self.device, generator=generator)
        z_coords = coordinates
        return torch.cat([z_basis, z_coords])
    
    def get_parameterization_info(self) -> dict:
        """Return information about the parameterization."""
        return {
            'k': self.k,
            'D': self.D,
            'd_total': self.d,
            'd_basis': self.d_basis,
            'd_coords': self.d_coords,
            'basis_parameterization': self.basis_parameterization,
            'basis_constraint': self.basis_constraint,
            'compression_ratio': self.D / self.k,  # Effective compression
            'parameterization_overhead': self.d_basis / (self.D * self.k)  # How much overhead
        }
    
    def analyze_basis_quality(self, z: Union[np.ndarray, torch.Tensor]) -> dict:
        """Analyze the quality of the basis encoded in the latent vector."""
        basis_matrix = self.get_basis_from_latent(z)
        
        # Compute basis quality metrics
        # 1. Orthogonality
        gram_matrix = basis_matrix.t() @ basis_matrix
        identity = torch.eye(self.k, device=self.device)
        orthogonality_error = (gram_matrix - identity).norm().item()
        
        # 2. Condition number
        singular_values = torch.linalg.svdvals(basis_matrix)
        condition_number = (singular_values.max() / (singular_values.min() + 1e-8)).item()
        
        # 3. Span coverage (how well distributed the basis vectors are)
        span_coverage = (singular_values > 1e-6).sum().item() / self.k
        
        return {
            'orthogonality_error': orthogonality_error,
            'condition_number': condition_number,
            'span_coverage': span_coverage,
            'singular_values': singular_values.tolist(),
            'basis_norm': basis_matrix.norm().item()
        }