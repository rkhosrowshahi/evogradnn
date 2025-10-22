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

    def __init__(self, model: torch.nn.Module, id: int, alpha: float = 1.0, device: str = 'cuda', seed: int = 0, train_biases: bool = True) -> None:
        """
        Initialize parameter sharing.

        Args:
            model: PyTorch model (e.g., ResNet-18).
            id: Number of intrinsic dimensions.
            alpha: Scaling factor for parameters.
            device: Device for PyTorch computations.
            seed: Random seed for reproducible initialize.
            train_biases: Whether to train bias parameters. If False, only weight parameters are trained.
        """
        super().__init__()
        self.device = device
        self.model = model
        self.seed = seed
        self.train_biases = train_biases
        
        theta_base = params_to_vector(self.model.parameters())
        self.register_buffer('theta_base', theta_base)
        self.D = len(theta_base)  # Full parameter count
        
        # Compute trainable parameter indices (excluding bias, batchnorm, etc. if train_biases=False)
        trainable_indices = self._compute_trainable_indices()
        self.register_buffer('trainable_indices', trainable_indices)
        self.N = len(trainable_indices)  # Number of trainable parameters
        
        self.id = id
        self.alpha = alpha
        self.other_d = 0

        if self.train_biases:
            self.other_d += self.num_biases

        self.d = self.id + self.other_d

    def _compute_trainable_indices(self) -> torch.Tensor:
        """
        Compute indices of trainable parameters based on train_biases flag.
        
        If train_biases=True, all parameters are trainable (indices = 0 to D-1).
        If train_biases=False, only weight parameters are trainable (excluding bias, batchnorm, layernorm, dropout).
        
        Returns:
            Tensor of trainable parameter indices
        """
        
        # Identify which parameters are weights (not bias, batchnorm, layernorm, dropout)
        trainable_mask = []
        offset = 0

        self.weight_indices = []
        self.bias_indices = []
        
        for name, param in self.model.named_parameters():
            param_size = param.numel()
            indices = list(range(offset, offset + param_size))
            
            # Check if this is a trainable weight parameter
            is_trainable = True
            
            # Exclude bias parameters
            is_bias = 'bias' in name
            
            # Also treat normalization layer parameters as "bias-like" (no projection)
            is_norm = any(norm in name for norm in ['bn', 'batch_norm', 'batchnorm', 'BatchNorm',
                                                      'ln', 'layer_norm', 'layernorm', 'LayerNorm',
                                                      'gn', 'group_norm', 'groupnorm', 'GroupNorm',
                                                      'instance_norm', 'instancenorm', 'InstanceNorm'])

            if is_bias or is_norm:
                is_trainable = False
            
            # Note: Dropout layers don't have parameters, so we don't need to check for them
            
            if is_trainable:
                # Add indices for this parameter
                trainable_mask.extend(range(offset, offset + param_size))
            
            if is_bias or is_norm:
                self.bias_indices.extend(indices)
            else:
                self.weight_indices.extend(indices)
            
            offset += param_size
        
        self.weight_indices = torch.tensor(self.weight_indices, dtype=torch.long, device=self.device)
        self.bias_indices = torch.tensor(self.bias_indices, dtype=torch.long, device=self.device)
        self.num_weights = len(self.weight_indices) # Total number of weight parameters
        self.num_biases = len(self.bias_indices) # Total number of biases and normalization layer parameters
        
        if self.train_biases:
            # All parameters are trainable
            return torch.arange(self.D, device=self.device)
        else:
            return torch.tensor(trainable_mask, dtype=torch.long, device=self.device)

    def set_theta(self, theta: Union[np.ndarray, torch.Tensor]) -> None:
        """Set the base parameter vector.
        
        Args:
            theta: Base parameter vector of shape (D,)
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.from_numpy(theta)
        
        if len(theta) == self.D:
            self.theta_base.copy_(theta)
        else:
            raise ValueError(f"Expected parameter vector of size {self.D}, got {len(theta)}")

    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using simple assignment strategy.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Expanded parameter vector of shape (D,)
        """
        NotImplementedError("This method is not implemented for the base class.")

    def reset(self, theta_base: Union[np.ndarray, torch.Tensor]) -> None:
        """Reset the weight sharing to initial state."""
        self.set_theta(theta_base)
        self.initialize()
    
    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to torch tensor on correct device."""
        if not isinstance(x, torch.Tensor):
            if hasattr(x, 'shape'):  # JAX array
                x = np.array(x)
            if isinstance(x, jax.Array):
                x = np.array(x)
            x = torch.from_numpy(x).float()
        return x.to(self.device)
    
    def _map_to_full_space(self, x_trainable: torch.Tensor) -> torch.Tensor:
        """
        Map trainable parameter vector to full parameter space.
        
        If train_biases=True, x_trainable is already of size D, so return as is.
        If train_biases=False, x_trainable is of size N, and we need to map it to size D
        by placing values at trainable_indices and zeros elsewhere.
        
        Args:
            x_trainable: Tensor of shape (N,) if train_biases=False, else (D,)
            
        Returns:
            Tensor of shape (D,) representing full parameter space
        """
        if self.train_biases:
            return x_trainable
        
        # Create zero vector of size D
        x_full = torch.zeros(self.D, device=self.device, dtype=x_trainable.dtype)
        # Place trainable values at their indices
        x_full[self.trainable_indices] = x_trainable
        return x_full
    
    def process(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Process expanded parameters with scaling and optional normalization.
        
        Args:
            x: Expanded parameter vector of shape (N,) if train_biases=False, else (D,)
            
        Returns:
            Processed parameter tensor of shape (D,) mapped to full space and added to theta_base
        """
        x = self._to_tensor(x)
        # Map to full parameter space (if train_biases=False, this maps N -> D)
        x_full = self._map_to_full_space(self.alpha * x)
        # Add to base parameters
        return self.theta_base + x_full
    
    def load_to_model(self, theta: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Load processed parameters into the neural network model.
        
        Args:
            theta: Parameter tensor of shape (D,) to load into model
        """
        theta = self._to_tensor(theta)
        if len(theta) == self.D:
            torch.nn.utils.vector_to_parameters(theta, self.model.parameters())
        else:
            raise ValueError(f"Expected parameter vector of size {self.D}, got {len(theta)}")
    

class RandomProjectionSoftSharing(ParameterSharing):
    """
    Parameter sharing using random projection matrices.
    
    This class implements soft parameter sharing by projecting from a low-dimensional
    latent space to the full parameter space using a random projection matrix P.
    
    When train_biases=True:
    - Weight parameters: theta_weights = theta_base_weights + P @ z_weights
    - Bias parameters: theta_bias = theta_base_bias + z_bias (1:1 identity mapping)
    - Total latent dimension = d + num_bias_params
    
    When train_biases=False:
    - Only weight parameters are trainable via projection
    - Bias parameters remain at theta_base values
    - Total latent dimension = d
    
    Input types:
        - Model: torch.nn.Module
        - d: int (latent dimension for weight parameters)
        - Scaling factor alpha: float
        - Normalization: bool (whether to normalize the projection matrix)  
    
    Output types:
        - Full parameter vector: torch.Tensor of shape (D,) where D is the
          total number of model parameters
    """
    
    def __init__(self, model: torch.nn.Module, d: int, alpha: float = 1.0, normalize: bool = False, device: str = 'cuda', seed: int = 0, train_biases: bool = True) -> None:
        super().__init__(model, id, alpha, device, seed, train_biases)
        self.normalize = normalize
        
        self.initialize()

    def initialize(self) -> None:
        """
        Initialize the random projection matrix with seed.
        
        Returns:
            Random projection matrix P of shape (num_weights, id)
        """
        # Projection matrix is always for weights only
            
        P = torch.randn(self.num_weights, self.id, device=self.device)
        if self.normalize:
            P, _ = torch.linalg.qr(P)
            P = P / (self.id ** 0.5)
        self.register_buffer('P', P)

    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using random projection.
        
        Args:
            z: Latent vector of shape (d,)
               If train_biases=True: z = [z_weights (id), z_biases (num_biases)]
               If train_biases=False: z = [z_weights (id)]
            
        Returns:
            Full parameter tensor of shape (D,) computed as theta_base + delta
        """
        z = self._to_tensor(z)
        
        if self.train_biases:
            # Split z into weight and bias components
            z_weights = z[:self.id]
            z_biases = z[self.id:]
            
            # Project weights
            delta_weights = self.P @ z_weights  # Shape: (self.num_weights,)
            
            # Create full delta vector
            delta = torch.zeros(self.D, device=self.device)
            delta[self.weight_indices] = delta_weights
            delta[self.bias_indices] = z_biases
            
            # Apply scaling and add to base
            return self.theta_base + self.alpha * delta
        else:
            # Only project trainable weights
            x = self.P @ z  # Shape: (self.num_weights,)
            # process() will map x to full space and add to theta_base
            return self.process(x)

    def latent_rotate(self, R: torch.Tensor) -> None:
        """Apply an orthogonal rotation in latent space: P <- P @ R.
        Args:
            R: Rotation matrix of shape (d, d), should be orthonormal.
        """
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, dtype=self.P.dtype, device=self.device)
        R = R.to(self.device)
        self.P.copy_(self.P @ R)


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
    
    def __init__(self, model: torch.nn.Module, id: int, 
                 sigma: float = 1.0, alpha: float = 1.0, device: str = 'cuda', seed: int = 0, train_biases: bool = True) -> None:
        """
        Initialize Random Fourier Features soft sharing.
        
        Args:
            model: PyTorch model (e.g., ResNet-18).
            id: Number of intrinsic dimensions.
            sigma: Standard deviation for omega sampling (1/sigma^2 is the variance).
            device: Device for PyTorch computations.
            seed: Random seed for reproducible initialize.
            train_biases: Whether to train bias parameters (1:1 mapping if True, frozen if False).
        """
        super().__init__(model, id, alpha, device, seed, train_biases)
        self.sigma = sigma
        
        self.initialize()

    def initialize(self) -> None:
        """
        Initialize Random Fourier Features parameters with seed.
        
        Draws omega from N(0, sigma^{-2} I) and b from uniform [0, 2π].
        """
        # RFF is always for weights only
        
        # Draw omegas from N(0, sigma^{-2} I)
        omega = torch.randn(self.num_weights, self.id, device=self.device) / self.sigma
        self.register_buffer('omega', omega)
        
        # Draw biases from uniform [0, 2π]
        b = torch.rand(self.num_weights, device=self.device) * 2 * np.pi
        self.register_buffer('b', b)

    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using Random Fourier Features.
        
        Args:
            z: Latent vector of shape (d,)
               If train_biases=True: z = [z_weights (id), z_biases (num_biases)]
               If train_biases=False: z = [z_weights (id)]
            
        Returns:
            Full parameter tensor of shape (D,)
        """
        z = self._to_tensor(z)
        
        if self.train_biases:
            # Split z into weight and bias components
            z_weights = z[:self.id]
            z_biases = z[self.id:]
            
            # Apply RFF to weights
            linear_combination = torch.matmul(self.omega, z_weights) + self.b  # Shape: (self.num_weights,)
            delta_weights = np.sqrt(2.0 / self.num_weights) * torch.cos(linear_combination)  # Shape: (self.num_weights,)
            
            # Create full delta vector
            delta = torch.zeros(self.D, device=self.device)
            delta[self.weight_indices] = delta_weights
            delta[self.bias_indices] = z_biases
            
            # Apply scaling and add to base
            return self.theta_base + self.alpha * delta
        else:
            # Apply RFF to trainable weights only
            linear_combination = torch.matmul(self.omega, z) + self.b  # Shape: (self.num_weights,)
            x = np.sqrt(2.0 / self.num_weights) * torch.cos(linear_combination)  # Shape: (self.num_weights,)
            # process() will apply alpha, map to full space, and add to theta_base
            return self.process(x)


class FastfoodProjection(ParameterSharing):
    """
    Parameter sharing using Fastfood transform - an efficient structured random projection.
    
    This class implements the Fastfood transform from Le et al. (2013) which provides
    fast and memory-efficient random projections using the Fast Walsh-Hadamard Transform.
    
    The mapping is: theta = theta_base + alpha * M @ z
    where M = H @ G @ Π @ B @ H is factored into:
    - H: Hadamard transform (via Fast Walsh-Hadamard Transform)
    - G: Random diagonal matrix with standard normal entries
    - Π: Random permutation
    - B: Random diagonal matrix with random ±1 entries
    
    Computational complexity: O(D log D) instead of O(D*d) for dense projections
    Memory complexity: O(D) instead of O(D*d)
    
    Note: D is automatically padded to the next power of 2 if needed for FWHT.
    
    Input types:
        - Model: torch.nn.Module
        - d: int (latent dimension)
        - alpha: float (scaling factor)
        
    Output types:
        - Full parameter vector: torch.Tensor of shape (D,) where D is the
          total number of model parameters
    
    Reference:
        Le, Q., Sarlós, T., & Smola, A. (2013). 
        Fastfood-approximating kernel expansions in loglinear time. ICML 2013.
    """
    
    def __init__(self, model: torch.nn.Module, id: int, alpha: float = 1.0, 
                 device: str = 'cuda', seed: int = 0, train_biases: bool = True) -> None:
        """
        Initialize Fastfood projection.
        
        Args:
            model: PyTorch model
            id: Number of intrinsic dimensions
            alpha: Scaling factor for the projection
            device: Device for computations ('cuda' or 'cpu')
            seed: Random seed for reproducibility
            train_biases: Whether to train bias parameters (1:1 mapping if True, frozen if False)
        """
        super().__init__(model, id, alpha, device, seed, train_biases)
        
        self.initialize()
    
    def initialize(self) -> None:
        """
        Initialize Fastfood transform parameters.
        
        Creates the fixed random matrices B, G, and permutation Π.
        Pads weight parameter count to the next power of 2 if needed for FWHT.
        """
        # Fastfood transform is always for weights only
        
        # Pad to next power of 2 for Fast Walsh-Hadamard Transform
        self.num_weights_padded = 2 ** int(np.ceil(np.log2(self.num_weights)))
        
        # Number of blocks needed to cover id dimensions
        # Each block has D_padded dimensions, so we need ceil(id / D_padded) blocks
        self.n_blocks = int(np.ceil(self.id / self.num_weights_padded))
        
        # For each block, create B, G, and permutation
        # B: Random diagonal matrix with ±1 entries
        B_list = []
        G_list = []
        perm_list = []
        
        for i in range(self.n_blocks):
            # B: Random ±1 diagonal matrix
            B = torch.randint(0, 2, (self.num_weights_padded,), device=self.device) * 2 - 1
            B = B.float()
            B_list.append(B)
            
            # G: Random diagonal matrix with standard normal entries
            G = torch.randn(self.num_weights_padded, device=self.device)
            G_list.append(G)
            
            # Π: Random permutation
            perm = torch.randperm(self.num_weights_padded, device=self.device)
            perm_list.append(perm)
        
        # Store as buffers (non-trainable)
        for i, (B, G, perm) in enumerate(zip(B_list, G_list, perm_list)):
            self.register_buffer(f'B_{i}', B)
            self.register_buffer(f'G_{i}', G)
            self.register_buffer(f'perm_{i}', perm)
        
        # Store scaling factor for normalization
        scaling = 1.0 / np.sqrt(self.num_weights_padded)
        self.register_buffer('scaling', torch.tensor(scaling, device=self.device))
        
        print(f"Fastfood initialized: D={self.num_weights}, num_weights_padded={self.num_weights_padded}, "
              f"id={self.id}, n_blocks={self.n_blocks}")
    
    def _fwht(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast Walsh-Hadamard Transform (vectorized implementation).
        
        Computes H @ x in O(D log D) time where H is a Hadamard matrix.
        
        Args:
            x: Input tensor of shape (D,) where D is a power of 2
            
        Returns:
            Transformed tensor of shape (D,)
        """
        # Vectorized implementation of Fast Walsh-Hadamard Transform
        n = x.shape[0]
        result = x.clone()
        h = 1
        
        while h < n:
            # Vectorized butterfly operations
            result = result.reshape(n // (2 * h), 2, h).transpose(0, 1)
            result = torch.cat([result[0] + result[1], result[0] - result[1]], dim=0)
            result = result.reshape(n)
            h *= 2
        
        # Normalize by sqrt(n) to maintain variance
        return result / np.sqrt(n)
    
    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Apply Fastfood transform to map latent vector to full parameter space.
        
        Computes: theta = theta_base + alpha * M @ z
        where M = H @ G @ Π @ B @ H (applied in blocks)
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,)
        """
        z = self._to_tensor(z)
        
        if self.train_biases:
            # Split z into weight and bias components
            z_weights = z[:self.id]
            z_biases = z[self.id:]
        else:
            z_weights = z
        
        # Initialize output
        output = torch.zeros(self.num_weights, device=self.device)
        
        # Apply Fastfood transform for each block
        for block_idx in range(self.n_blocks):
            # Get the portion of z_weights for this block
            start_idx = block_idx * self.D_padded
            end_idx = min(start_idx + self.D_padded, self.id)
            
            # Pad z_block to D_padded if needed
            z_block = torch.zeros(self.num_weights, device=self.device)
            z_block[:end_idx - start_idx] = z_weights[start_idx:end_idx]
            
            # Get block-specific matrices
            B = getattr(self, f'B_{block_idx}')
            G = getattr(self, f'G_{block_idx}')
            perm = getattr(self, f'perm_{block_idx}')
            
            # Apply Fastfood transform: H @ G @ Π @ B @ H @ z
            # Step 1: H @ z_block (First Hadamard transform)
            x = self._fwht(z_block)
            
            # Step 2: B @ x (Element-wise multiplication with ±1 diagonal)
            x = B * x
            
            # Step 3: Π @ x (Permutation)
            x = x[perm]
            
            # Step 4: G @ x (Element-wise multiplication with Gaussian diagonal)
            x = G * x
            
            # Step 5: H @ x (Second Hadamard transform)
            x = self._fwht(x)
            
            # Accumulate to output
            output += x
        
        # Scale and truncate to num_weights
        output = output[:self.num_weights] * self.scaling
        
        if self.train_biases:
            # Create full delta vector
            delta = torch.zeros(self.D, device=self.device)  # Trainable parameters Shape: (D,)
            delta[self.weight_indices] = output
            delta[self.bias_indices] = z_biases
            
            # Apply scaling and add to base
            return self.theta_base + self.alpha * delta  # Shape: (num_weights,)
        else:
            # Process will map to full space and add to base parameters
            return self.process(output)  # Shape: (num_weights,)


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
    
    def __init__(self, model: torch.nn.Module, id: int, 
                 hidden_dims: List[int] = [32, 16], use_activation: bool = True, alpha: float = 1.0,
                 activation: str = 'relu', device: str = 'cuda', seed: int = 0) -> None:
        super().__init__(model, id, alpha, device, seed)

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
        self.initialize()

    def initialize(self) -> None:
        """
        Initialize decoder weights using He normal initialize.
        
        Each layer is initialized with standard deviation sqrt(1/fan_in)
        to maintain reasonable activation magnitudes.
        """
        layers = []
        # First layer: input to first hidden
        layer = nn.Linear(self.id, self.hidden_dims[0], bias=False)
        # Normalize weights to unit norm
        with torch.no_grad():
            layer.weight.data = layer.weight.data / torch.norm(layer.weight.data, dim=1, keepdim=True)
            layer.weight.data = layer.weight.data / self.id ** 0.5
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
        - Embedding initialize: str (embedding initialize)
    
    Output types:
        - Full parameter vector: torch.Tensor of shape (D,) where D is the
          total number of model parameters across all layers
    """
    
    def __init__(self, model: torch.nn.Module, id: int, 
                 hidden_dims: List[int] = [32, 16], emb_init: str = 'sinusoidal', alpha: float = 1.0,
                 device: str = 'cuda', seed: int = 0) -> None:
        super().__init__(model, id, alpha, device, seed)

        self.hidden_dims = hidden_dims
        
        layer_sizes = []
        for layer in model.parameters():
            layer_sizes.append(layer.numel())

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.embed_dim = 16

        if emb_init == "random":
            # Option 1: Random initialize with seed
            embeddings = torch.randn(self.num_layers, self.embed_dim, device=device) * 0.1
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
            nn.Linear(self.id + self.embed_dim, hidden_dims[0]),
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
        if self.train_biases:
            # Create full delta vector
            delta = torch.zeros(self.D, device=self.device)  # Trainable parameters Shape: (D,)
            delta[self.weight_indices] = x
            delta[self.bias_indices] = z[self.id:]
            
            # Apply scaling and add to base
            return self.theta_base + self.alpha * delta  # Shape: (num_weights,)
        else:
            return self.process(x)


class SparseProjection(ParameterSharing):
    """Sparse random projection: P is sparse matrix with density 'sparsity'.
    Non-zeros ~ N(0,1), then columns normalized to unit length.
    Projection applied only to weight parameters, biases use 1:1 mapping if train_biases=True.
    """
    def __init__(self, model: torch.nn.Module, id: int, 
                 sparsity: float = 0.1, alpha: float = 1.0, device: str = 'cuda', seed: int = 0, train_biases: bool = True):
        super().__init__(model, id, alpha, device, seed, train_biases)
        self.sparsity = sparsity
        
        # Projection matrix is always for weights only
        
        # Generate indices for non-zeros: random positions with seed
        num_nz = int(self.num_weights * self.id * self.sparsity)
        row_idx = torch.randint(0, self.num_weights, (num_nz,), device=device)
        col_idx = torch.randint(0, self.id, (num_nz,), device=device)
        
        # Values: N(0,1)
        values = torch.randn(num_nz, device=device)
        
        # Sparse tensor in COO
        P_coo = torch.sparse_coo_tensor(
            torch.stack([row_idx, col_idx]), values,
            size=(self.num_weights, self.id), device=device
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
               If train_biases=True: z = [z_weights (id), z_biases (num_biases)]
               If train_biases=False: z = [z_weights (id)]
            
        Returns:
            Full parameter tensor of shape (D,)
        """
        z = self._to_tensor(z)
        
        if self.train_biases:
            # Split z into weight and bias components
            z_weights = z[:self.id]
            z_biases = z[self.id:]
            
            # Convert to dense for matmul
            P_dense = self.P.to_dense()
            delta_weights = P_dense @ z_weights
            
            # Create full delta vector
            delta = torch.zeros(self.D, device=self.device)
            delta[self.weight_indices] = delta_weights
            delta[self.bias_indices] = z_biases
            
            # Apply scaling and add to base
            return self.theta_base + self.alpha * delta
        else:
            # Convert to dense for matmul
            P_dense = self.P.to_dense()
            x = P_dense @ z  # Shape: (N,)
            # process() will map to full space and add to theta_base
            return self.process(x)




class HardWeightSharing(ParameterSharing):
    """
    Args:
        model: PyTorch model
        d: Number of blocks (latent dimension)
        alpha: Scaling factor for parameters
        device: Device for computations
        assignment_strategy: How to initialize assignments ('random', 'uniform')
    """
    
    def __init__(self, model: torch.nn.Module, id: int, seed: int = 0,
                 alpha: float = 1.0, device: str = 'cuda', 
                 assignment_strategy: str = 'random', train_biases: bool = True) -> None:
        super().__init__(model, id, alpha, device, seed, train_biases)
        self.assignment_strategy = assignment_strategy
        
        self.initialize()
        
        print(f"HardWeightSharing initialized:")
        print(f"  - Number of weights: {self.num_weights}")
        print(f"  - Number of blocks: {self.id}")
        print(f"  - Total parameters: {self.D}")
        print(f"  - Compression ratio: {self.get_compression_ratio():.2f}")
        print(self.count_block_sizes())
    
    def initialize(self) -> None:
        """Initialize parameter assignments to blocks."""
        if self.assignment_strategy == 'random':
            # Random assignment with seed
            assignments = torch.randint(0, self.id, (self.num_weights,), device=self.device)
        elif self.assignment_strategy == 'uniform':
            # Uniform distribution across blocks
            assignments = torch.arange(self.num_weights, device=self.device) % self.id
        else:
            raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy}")
        
        # Store assignments as buffer (non-learnable)
        self.register_buffer('assignments', assignments)
    
    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector (blocks) to full parameter space using hard assignment.
        
        Args:
            z: Blocks vector of shape (d,) - these become the blocks
            
        Returns:
            Full parameter tensor of shape (D,) where each position gets the value
            from the corresponding block: theta[i] = z[assignments[i]]
        """
        z = self._to_tensor(z)
        
        # Use the input z as the blocks values and look up based on assignments
        if self.train_biases:
            z_weights = z[:self.id]
            z_biases = z[self.id:]
        else:
            z_weights = z
        output = z_weights[self.assignments]

        if self.train_biases:
            # Create full delta vector
            delta = torch.zeros(self.D, device=self.device)  # Trainable parameters Shape: (D,)
            delta[self.weight_indices] = output
            delta[self.bias_indices] = z_biases
            
            # Apply scaling and add to base
            return self.theta_base + self.alpha * delta  # Shape: (N,)
        else:
            # Process will map to full space and add to base parameters
            return self.process(output)  # Shape: (num_weights,)
    
    def count_block_sizes(self) -> torch.Tensor:
        """Return how many parameters are assigned to each block."""
        usage = torch.zeros(self.id, device=self.device)
        for i in range(self.id):
            usage[i] = (self.assignments == i).sum()
        return usage
    
    def get_compression_ratio(self) -> float:
        """Return the compression ratio achieved by the blocks."""
        return self.D / self.d


class LayerwiseHardWeightSharing(ParameterSharing):
    """
    Layerwise Hard Weight Sharing: Each layer has its own set of blocks.
    
    Unlike HardWeightSharing where all parameters share d blocks, this class
    organizes blocks by layer. Each weight layer gets k blocks, and parameters within
    a layer can only be assigned to that layer's blocks.
    
    For bias parameters, the number of blocks is adaptively set to min(k, bias_size)
    to avoid having more blocks than parameters (which would be wasteful).
    
    Total number of blocks = sum of blocks across all layers, where:
    - Weight layers get k blocks each
    - Bias layers get min(k, bias_size) blocks each
    
    Args:
        model: PyTorch model
        d: Number of blocks per weight layer (k)
        seed: Random seed for reproducible assignment generation
        alpha: Scaling factor for parameters
        device: Device for computations
        assignment_strategy: How to initialize assignments within each layer ('random', 'uniform')
    """
    
    def __init__(self, model: torch.nn.Module, id: int, seed: int = 0,
                 alpha: float = 1.0, device: str = 'cuda', 
                 assignment_strategy: str = 'random', train_biases: bool = True) -> None:

        super().__init__(model, id, alpha, device, seed, train_biases)
        
        self.k = id  # Blocks per layer
        self.assignment_strategy = assignment_strategy
        
        # Extract layer information
        self._extract_layer_info()
        
        # Initialize layer-wise assignments
        self.initialize()
        
        print(f"LayerwiseHardWeightSharing initialized:")
        print(f"  - Number of layers: {self.num_layers}")
        print(f"  - Blocks per layer (k): {self.k}")
        print(f"  - Total blocks (id): {self.id}")
        print(f"  - Total parameters (D): {self.D}")
        print(f"  - Layers info: {self.layer_info}")
        print(f"  - Compression ratio: {self.get_compression_ratio():.2f}")
        print(self.count_block_sizes())
    
    def _extract_layer_info(self) -> None:
        """Extract layer boundaries and information from the model."""
        self.layer_boundaries = []  # List of (start_idx, end_idx) for each layer
        self.layer_sizes = []  # Number of parameters in each layer
        self.layer_blocks = []  # Number of blocks for each layer
        self.layer_info = {}  # Layer information
        
        start_idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layer_size = param.numel()
                end_idx = start_idx + layer_size
                
                # For bias parameters, use min(k, bias_size) blocks to avoid having more blocks than parameters
                is_bias = 'bias' in name
                num_blocks = min(self.k, layer_size) if is_bias else self.k
                
                self.layer_boundaries.append((start_idx, end_idx))
                self.layer_sizes.append(layer_size)
                self.layer_blocks.append(num_blocks)
                self.layer_info[name] = {'size': layer_size, 'blocks': num_blocks, 'is_bias': is_bias}
                
                start_idx = end_idx
        
        self.num_layers = len(self.layer_boundaries)
        # Total blocks = sum of blocks across all layers (not just k * num_layers)
        self.id = sum(self.layer_blocks)
        
    def initialize(self) -> None:
        """Initialize parameter assignments to layer-specific blocks."""
        assignments = torch.zeros(self.D, dtype=torch.long, device=self.device)
        
        # Calculate cumulative block indices for each layer
        block_offsets = [0]
        for num_blocks in self.layer_blocks:
            block_offsets.append(block_offsets[-1] + num_blocks)
        
        for layer_idx, (start, end) in enumerate(self.layer_boundaries):
            layer_size = end - start
            num_layer_blocks = self.layer_blocks[layer_idx]
            
            # Block indices for this layer
            layer_block_start = block_offsets[layer_idx]
            layer_block_end = block_offsets[layer_idx + 1]
            
            if self.assignment_strategy == 'random':
                # Random assignment within layer's blocks
                layer_assignments = torch.randint(
                    layer_block_start, 
                    layer_block_end, 
                    (layer_size,), 
                    device=self.device, 
                )
            elif self.assignment_strategy == 'uniform':
                # Uniform distribution across layer's blocks
                layer_assignments = torch.arange(layer_size, device=self.device) % num_layer_blocks
                layer_assignments += layer_block_start
            else:
                raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy}")
            
            assignments[start:end] = layer_assignments
        
        # Store block offsets for later use
        self.register_buffer('block_offsets', torch.tensor(block_offsets, device=self.device))
        # Store assignments as buffer (non-learnable)
        self.register_buffer('assignments', assignments)
        
        # Recalculate d based on actually used blocks (important for random assignment)
        self._recalculate_d_from_assignments()
    
    def _recalculate_d_from_assignments(self) -> None:
        """
        Recalculate self.d based on unique blocks actually used in assignments.
        
        With random assignment, some blocks may not be assigned to any parameters.
        This method counts unique blocks per layer, remaps assignments to be contiguous,
        and updates self.d to reflect the actual latent space dimensionality.
        """
        new_assignments = torch.zeros_like(self.assignments)
        new_layer_blocks = []
        new_block_offsets = [0]
        global_block_idx = 0
        
        for layer_idx, (start, end) in enumerate(self.layer_boundaries):
            # Get assignments for this layer
            layer_assignments = self.assignments[start:end]
            
            # Find unique blocks used in this layer
            unique_blocks = torch.unique(layer_assignments)
            num_unique = len(unique_blocks)
            
            # Create mapping from old block indices to new contiguous indices
            old_to_new = {}
            for new_idx, old_block in enumerate(unique_blocks):
                old_to_new[old_block.item()] = global_block_idx + new_idx
            
            # Remap assignments for this layer
            for i in range(start, end):
                old_block = self.assignments[i].item()
                new_assignments[i] = old_to_new[old_block]
            
            # Update tracking
            new_layer_blocks.append(num_unique)
            global_block_idx += num_unique
            new_block_offsets.append(global_block_idx)
            
            # Update layer_info with actual blocks used
            layer_name = list(self.layer_info.keys())[layer_idx]
            old_blocks = self.layer_info[layer_name]['blocks']
            self.layer_info[layer_name]['blocks'] = num_unique
            
            if num_unique < old_blocks:
                print(f"  Layer '{layer_name}': {old_blocks} blocks allocated, {num_unique} actually used")
        
        # Update all relevant attributes
        old_d = self.d
        self.d = global_block_idx
        self.layer_blocks = new_layer_blocks
        self.assignments = new_assignments
        self.register_buffer('block_offsets', torch.tensor(new_block_offsets, device=self.device))
        
        if self.d < old_d:
            print(f"  Total blocks reduced from {old_d} to {self.d} (removed {old_d - self.d} unused blocks)")
    
    def reset(self, theta_base: Union[np.ndarray, torch.Tensor]) -> None:
        """Reset the weight sharing to initial state."""
        self.set_theta(theta_base)
        # Do not reset assignments due to change in dimensions.
        # TODO: without dimension change, reset assignments. OR reset assignments and reset d in optimizer. OR update assignment with clustering and again reset d in optimizer.

    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector (blocks) to full parameter space using layerwise hard assignment.
        
        Args:
            z: Blocks vector of shape (d,) where d = sum of blocks across all layers.
               The blocks are organized sequentially by layer:
               - First num_blocks[0] elements are blocks for layer 0
               - Next num_blocks[1] elements are blocks for layer 1
               - etc.
               For weight layers: num_blocks = k
               For bias layers: num_blocks = min(k, bias_size)
            
        Returns:
            Full parameter tensor of shape (D,) where each position gets the value
            from its assigned block: theta[i] = z[assignments[i]]
        """
        z = self._to_tensor(z)
        
        # Validate input dimension
        if z.shape[0] != self.d:
            raise ValueError(f"Expected latent vector of size {self.d}, got {z.shape[0]}")
        
        # Use the input z as the blocks values and look up based on assignments
        x = z[self.assignments]
        # Hard weight sharing already produces full D-dimensional output
        # Just apply alpha scaling directly
        return self.theta_base + self.alpha * x
    
    def count_block_sizes(self) -> dict:
        """Return how many parameters are assigned to each block, organized by layer."""
        block_usage = {}
        
        for layer_idx, (layer_name, layer_data) in enumerate(self.layer_info.items()):
            num_layer_blocks = self.layer_blocks[layer_idx]
            layer_start_block = self.block_offsets[layer_idx].item()
            layer_end_block = self.block_offsets[layer_idx + 1].item()
            
            layer_usage = torch.zeros(num_layer_blocks, device=self.device)
            for local_block_idx in range(num_layer_blocks):
                global_block_idx = layer_start_block + local_block_idx
                layer_usage[local_block_idx] = (self.assignments == global_block_idx).sum()
            
            block_usage[layer_name] = layer_usage.cpu().numpy()
        
        return block_usage
    
    def get_compression_ratio(self) -> float:
        """Return the compression ratio achieved by the blocks."""
        return self.D / self.d
    
    def get_layer_info(self) -> dict:
        """Return detailed information about layer structure."""
        info = {
            'num_layers': self.num_layers,
            'blocks_per_layer_k': self.k,  # Target blocks per weight layer
            'total_blocks': self.d,
            'total_parameters': self.D,
            'compression_ratio': self.get_compression_ratio(),
            'layers': []
        }
        
        for layer_idx, (start, end) in enumerate(self.layer_boundaries):
            layer_size = end - start
            num_layer_blocks = self.layer_blocks[layer_idx]
            layer_block_start = self.block_offsets[layer_idx].item()
            layer_block_end = self.block_offsets[layer_idx + 1].item()
            
            layer_info = {
                'layer_idx': layer_idx,
                'param_range': (start, end),
                'param_size': layer_size,
                'block_range': (layer_block_start, layer_block_end),
                'num_blocks': num_layer_blocks,
                'layer_compression_ratio': layer_size / num_layer_blocks
            }
            info['layers'].append(layer_info)
        
        return info
    
    def get_blocks_for_layer(self, layer_idx: int, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract the block values for a specific layer from the latent vector."""
        z = self._to_tensor(z)
        layer_block_start = self.block_offsets[layer_idx].item()
        layer_block_end = self.block_offsets[layer_idx + 1].item()
        return z[layer_block_start:layer_block_end]
    
    def set_blocks_for_layer(self, layer_idx: int, z: torch.Tensor, 
                            layer_blocks: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Update the block values for a specific layer in the latent vector."""
        layer_blocks = self._to_tensor(layer_blocks)
        z_new = z.clone()
        layer_block_start = self.block_offsets[layer_idx].item()
        layer_block_end = self.block_offsets[layer_idx + 1].item()
        z_new[layer_block_start:layer_block_end] = layer_blocks
        return z_new


class LayerwiseHardWeightSharingV2(ParameterSharing):
    """
    Layerwise Hard Weight Sharing V2: Weight layers get individual blocks, all biases share blocks.
    
    This version differs from V1 by grouping all bias parameters together into a single set of k blocks,
    while each weight layer still gets its own k blocks.
    
    Structure:
    - Weight layer 1: k blocks
    - Weight layer 2: k blocks
    - ...
    - Weight layer L: k blocks
    - ALL biases combined: k blocks
    
    Total number of blocks = (num_weight_layers * k) + k
    
    Args:
        model: PyTorch model
        d: Number of blocks per layer (k)
        seed: Random seed for reproducible assignment generation
        alpha: Scaling factor for parameters
        device: Device for computations
        assignment_strategy: How to initialize assignments within each layer ('random', 'uniform')
    """
    
    def __init__(self, model: torch.nn.Module, d: int, seed: int = 0,
                 alpha: float = 1.0, device: str = 'cuda', 
                 assignment_strategy: str = 'random', train_biases: bool = True) -> None:

        super().__init__(model, d, alpha, device, seed, train_biases)
        
        self.k = d  # Blocks per layer
        self.assignment_strategy = assignment_strategy
        
        # Extract layer information (separating weights and biases)
        self._extract_layer_info()
        
        # Initialize layer-wise assignments
        self.initialize()
        
        print(f"LayerwiseHardWeightSharingV2 initialized:")
        print(f"  - Number of weight layers: {self.num_weight_layers}")
        print(f"  - Number of bias parameters: {self.num_bias_params}")
        print(f"  - Total bias size: {self.total_bias_size}")
        print(f"  - Blocks per layer (k): {self.k}")
        print(f"  - Total blocks (d): {self.d}")
        print(f"  - Total parameters (D): {self.D}")
        print(f"  - Compression ratio: {self.get_compression_ratio():.2f}")
        print(self.count_block_sizes())
    
    def _extract_layer_info(self) -> None:
        """Extract layer boundaries, separating weights and biases."""
        self.weight_boundaries = []  # List of (start_idx, end_idx) for weight layers
        self.weight_sizes = []  # Number of parameters in each weight layer
        self.weight_layer_names = []  # Names of weight layers
        
        self.bias_boundaries = []  # List of (start_idx, end_idx) for bias layers
        self.bias_sizes = []  # Number of parameters in each bias layer
        self.bias_layer_names = []  # Names of bias layers
        
        start_idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layer_size = param.numel()
                end_idx = start_idx + layer_size
                
                is_bias = 'bias' in name
                
                if is_bias:
                    self.bias_boundaries.append((start_idx, end_idx))
                    self.bias_sizes.append(layer_size)
                    self.bias_layer_names.append(name)
                else:
                    self.weight_boundaries.append((start_idx, end_idx))
                    self.weight_sizes.append(layer_size)
                    self.weight_layer_names.append(name)
                
                start_idx = end_idx
        
        self.num_weight_layers = len(self.weight_boundaries)
        self.num_bias_params = len(self.bias_boundaries)
        self.total_bias_size = sum(self.bias_sizes)
        
        # Total blocks = (num_weight_layers * k) + k (for all biases)
        self.d = self.num_weight_layers * self.k + self.k
        
        # Store layer information for easier access
        self.layer_info = {
            'weights': {
                name: {'size': size, 'blocks': self.k, 'is_bias': False}
                for name, size in zip(self.weight_layer_names, self.weight_sizes)
            },
            'biases': {
                'all_biases': {'size': self.total_bias_size, 'blocks': self.k, 'is_bias': True}
            }
        }
        
    def initialize(self) -> None:
        """Initialize parameter assignments to layer-specific blocks."""
        assignments = torch.zeros(self.D, dtype=torch.long, device=self.device)
        
        # Block offset tracking
        # First blocks are for weight layers, last k blocks are for all biases
        current_block_offset = 0
        
        # Assign weight layers
        for layer_idx, (start, end) in enumerate(self.weight_boundaries):
            layer_size = end - start
            layer_block_start = current_block_offset
            layer_block_end = current_block_offset + self.k
            
            if self.assignment_strategy == 'random':
                # Balanced random assignment: ensure all blocks are used
                layer_assignments = self._balanced_random_assignment(
                    layer_size, layer_block_start, layer_block_end
                )
            elif self.assignment_strategy == 'uniform':
                # Uniform distribution across layer's blocks
                layer_assignments = torch.arange(layer_size, device=self.device) % self.k
                layer_assignments += layer_block_start
            else:
                raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy}")
            
            assignments[start:end] = layer_assignments
            current_block_offset += self.k
        
        # Assign all biases to the last k blocks
        bias_block_start = current_block_offset
        bias_block_end = current_block_offset + self.k
        
        # Collect all bias parameters together
        all_bias_params = []
        for start, end in self.bias_boundaries:
            all_bias_params.append(torch.arange(start, end, device=self.device))
        
        if all_bias_params:
            all_bias_indices = torch.cat(all_bias_params)
            total_bias_size = len(all_bias_indices)
            
            if self.assignment_strategy == 'random':
                # Balanced random assignment: ensure all blocks are used
                bias_assignments = self._balanced_random_assignment(
                    total_bias_size, bias_block_start, bias_block_end
                )
                # Assign to the actual bias positions
                for i, idx in enumerate(all_bias_indices):
                    assignments[idx] = bias_assignments[i]
            elif self.assignment_strategy == 'uniform':
                # Uniform distribution across bias blocks
                bias_assignments = torch.arange(total_bias_size, device=self.device) % self.k
                bias_assignments += bias_block_start
                # Assign to the actual bias positions
                for i, idx in enumerate(all_bias_indices):
                    assignments[idx] = bias_assignments[i]
            else:
                raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy}")
        
        # Store block offsets for weights and biases
        weight_block_offsets = list(range(0, self.num_weight_layers * self.k + 1, self.k))
        bias_block_offset = self.num_weight_layers * self.k
        
        self.register_buffer('weight_block_offsets', 
                            torch.tensor(weight_block_offsets, device=self.device))
        self.register_buffer('bias_block_offset', 
                            torch.tensor(bias_block_offset, device=self.device))
        
        # Store assignments as buffer (non-learnable)
        self.register_buffer('assignments', assignments)
    
    def _balanced_random_assignment(self, num_params: int, block_start: int, block_end: int) -> torch.Tensor:
        """
        Create balanced random assignments that ensure all blocks are used.
        
        Strategy:
        1. First, assign one parameter to each block (round-robin)
        2. Then, randomly shuffle the assignments
        
        This guarantees all blocks get at least one parameter.
        
        Args:
            num_params: Number of parameters to assign
            block_start: Starting block index
            block_end: Ending block index (exclusive)
            
        Returns:
            Tensor of assignments of shape (num_params,)
        """
        num_blocks = block_end - block_start
        
        if num_params < num_blocks:
            # If we have fewer parameters than blocks, just assign sequentially
            assignments = torch.arange(block_start, block_start + num_params, device=self.device)
        else:
            # Create assignments that cycle through all blocks
            # This ensures each block gets at least floor(num_params / num_blocks) parameters
            assignments = torch.arange(num_params, device=self.device) % num_blocks
            assignments += block_start
            
            # Shuffle to make it random while keeping all blocks used
            perm = torch.randperm(num_params, device=self.device)
            assignments = assignments[perm]
        
        return assignments
    
    def _recalculate_d_from_assignments(self) -> None:
        """
        Recalculate self.d based on unique blocks actually used in assignments.
        
        With random assignment, some blocks may not be assigned to any parameters.
        This method counts unique blocks used and remaps assignments to be contiguous.
        """
        new_assignments = torch.zeros_like(self.assignments)
        global_block_idx = 0
        
        # Process weight layers
        new_weight_block_offsets = [0]
        actual_weight_blocks = []
        
        for layer_idx, (start, end) in enumerate(self.weight_boundaries):
            layer_block_start = self.weight_block_offsets[layer_idx].item()
            layer_block_end = self.weight_block_offsets[layer_idx + 1].item()
            
            # Get assignments for this weight layer
            layer_assignments = self.assignments[start:end]
            
            # Find unique blocks used in this layer (filtered to this layer's range)
            layer_mask = (layer_assignments >= layer_block_start) & (layer_assignments < layer_block_end)
            unique_blocks = torch.unique(layer_assignments[layer_mask])
            num_unique = len(unique_blocks)
            
            # Create mapping from old block indices to new contiguous indices
            old_to_new = {}
            for new_idx, old_block in enumerate(unique_blocks):
                old_to_new[old_block.item()] = global_block_idx + new_idx
            
            # Remap assignments for this layer
            for i in range(start, end):
                old_block = self.assignments[i].item()
                if old_block in old_to_new:
                    new_assignments[i] = old_to_new[old_block]
            
            # Update tracking
            actual_weight_blocks.append(num_unique)
            global_block_idx += num_unique
            new_weight_block_offsets.append(global_block_idx)
            
            if num_unique < self.k:
                layer_name = self.weight_layer_names[layer_idx]
                print(f"  Weight layer '{layer_name}': {self.k} blocks allocated, {num_unique} actually used")
        
        # Process all biases together
        bias_block_start = self.bias_block_offset.item()
        bias_block_end = bias_block_start + self.k
        
        # Collect all bias assignments
        all_bias_assignments = []
        for start, end in self.bias_boundaries:
            all_bias_assignments.append(self.assignments[start:end])
        
        if all_bias_assignments:
            all_bias_assignments = torch.cat(all_bias_assignments)
            
            # Find unique blocks used in biases
            bias_mask = (all_bias_assignments >= bias_block_start) & (all_bias_assignments < bias_block_end)
            unique_bias_blocks = torch.unique(all_bias_assignments[bias_mask])
            num_unique_bias = len(unique_bias_blocks)
            
            # Create mapping for bias blocks
            old_to_new_bias = {}
            for new_idx, old_block in enumerate(unique_bias_blocks):
                old_to_new_bias[old_block.item()] = global_block_idx + new_idx
            
            # Remap bias assignments
            for start, end in self.bias_boundaries:
                for i in range(start, end):
                    old_block = self.assignments[i].item()
                    if old_block in old_to_new_bias:
                        new_assignments[i] = old_to_new_bias[old_block]
            
            new_bias_block_offset = global_block_idx
            global_block_idx += num_unique_bias
            
            if num_unique_bias < self.k:
                print(f"  All biases: {self.k} blocks allocated, {num_unique_bias} actually used")
        else:
            new_bias_block_offset = global_block_idx
            num_unique_bias = 0
        
        # Update all relevant attributes
        old_d = self.d
        self.d = global_block_idx
        self.actual_weight_blocks = actual_weight_blocks
        self.actual_bias_blocks = num_unique_bias
        self.assignments = new_assignments
        self.register_buffer('weight_block_offsets', 
                            torch.tensor(new_weight_block_offsets, device=self.device))
        self.register_buffer('bias_block_offset', 
                            torch.tensor(new_bias_block_offset, device=self.device))
        
        if self.d < old_d:
            print(f"  Total blocks reduced from {old_d} to {self.d} (removed {old_d - self.d} unused blocks)")
    
    # def reset(self, theta_base: Union[np.ndarray, torch.Tensor]) -> None:
    #     """Reset the weight sharing to initial state."""
    #     self.set_theta(theta_base)
    #     # Do not reset assignments due to change in dimensions.

    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector (blocks) to full parameter space.
        
        Args:
            z: Blocks vector of shape (d,) where d = (num_weight_layers * k) + k
               The blocks are organized as:
               - First k blocks: weight layer 0
               - Next k blocks: weight layer 1
               - ...
               - Last k blocks: all biases
            
        Returns:
            Full parameter tensor of shape (D,) where each position gets the value
            from its assigned block: theta[i] = z[assignments[i]]
        """
        z = self._to_tensor(z)
        
        # Validate input dimension
        if z.shape[0] != self.d:
            raise ValueError(f"Expected latent vector of size {self.d}, got {z.shape[0]}")
        
        # Use the input z as the blocks values and look up based on assignments
        x = z[self.assignments]
        # Hard weight sharing already produces full D-dimensional output
        # Just apply alpha scaling directly
        return self.theta_base + self.alpha * x
    
    def count_block_sizes(self) -> dict:
        """Return how many parameters are assigned to each block, organized by layer."""
        block_usage = {}
        
        # Count for weight layers
        for layer_idx, layer_name in enumerate(self.weight_layer_names):
            start, end = self.weight_boundaries[layer_idx]
            layer_block_start = self.weight_block_offsets[layer_idx].item()
            layer_block_end = self.weight_block_offsets[layer_idx + 1].item()
            num_layer_blocks = layer_block_end - layer_block_start
            
            layer_usage = torch.zeros(num_layer_blocks, device=self.device)
            for local_block_idx in range(num_layer_blocks):
                global_block_idx = layer_block_start + local_block_idx
                layer_usage[local_block_idx] = (self.assignments[start:end] == global_block_idx).sum()
            
            block_usage[layer_name] = layer_usage.cpu().numpy()
        
        # Count for all biases together
        if self.bias_boundaries:
            bias_block_start = self.bias_block_offset.item()
            bias_block_end = self.d  # All remaining blocks are for biases
            num_bias_blocks = bias_block_end - bias_block_start
            
            bias_usage = torch.zeros(num_bias_blocks, device=self.device)
            for local_block_idx in range(num_bias_blocks):
                global_block_idx = bias_block_start + local_block_idx
                # Count across all bias parameters
                count = 0
                for start, end in self.bias_boundaries:
                    count += (self.assignments[start:end] == global_block_idx).sum()
                bias_usage[local_block_idx] = count
            
            block_usage['all_biases'] = bias_usage.cpu().numpy()
        
        return block_usage
    
    def get_compression_ratio(self) -> float:
        """Return the compression ratio achieved by the blocks."""
        return self.D / self.d
    
    def get_layer_info(self) -> dict:
        """Return detailed information about layer structure."""
        info = {
            'num_weight_layers': self.num_weight_layers,
            'num_bias_params': self.num_bias_params,
            'total_bias_size': self.total_bias_size,
            'blocks_per_layer_k': self.k,
            'total_blocks': self.d,
            'total_parameters': self.D,
            'compression_ratio': self.get_compression_ratio(),
            'weight_layers': [],
            'bias_info': None
        }
        
        # Weight layers info
        for layer_idx, layer_name in enumerate(self.weight_layer_names):
            start, end = self.weight_boundaries[layer_idx]
            layer_size = end - start
            layer_block_start = self.weight_block_offsets[layer_idx].item()
            layer_block_end = self.weight_block_offsets[layer_idx + 1].item()
            num_layer_blocks = layer_block_end - layer_block_start
            
            layer_info = {
                'name': layer_name,
                'layer_idx': layer_idx,
                'param_range': (start, end),
                'param_size': layer_size,
                'block_range': (layer_block_start, layer_block_end),
                'num_blocks': num_layer_blocks,
                'layer_compression_ratio': layer_size / num_layer_blocks if num_layer_blocks > 0 else 0
            }
            info['weight_layers'].append(layer_info)
        
        # Bias info
        if self.bias_boundaries:
            bias_block_start = self.bias_block_offset.item()
            bias_block_end = self.d
            num_bias_blocks = bias_block_end - bias_block_start
            
            info['bias_info'] = {
                'total_size': self.total_bias_size,
                'num_bias_params': self.num_bias_params,
                'block_range': (bias_block_start, bias_block_end),
                'num_blocks': num_bias_blocks,
                'compression_ratio': self.total_bias_size / num_bias_blocks if num_bias_blocks > 0 else 0
            }
        
        return info
    
    def get_blocks_for_weight_layer(self, layer_idx: int, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract the block values for a specific weight layer from the latent vector."""
        z = self._to_tensor(z)
        layer_block_start = self.weight_block_offsets[layer_idx].item()
        layer_block_end = self.weight_block_offsets[layer_idx + 1].item()
        return z[layer_block_start:layer_block_end]
    
    def get_blocks_for_biases(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract the block values for all biases from the latent vector."""
        z = self._to_tensor(z)
        bias_block_start = self.bias_block_offset.item()
        return z[bias_block_start:]
    
    def set_blocks_for_weight_layer(self, layer_idx: int, z: torch.Tensor, 
                                   layer_blocks: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Update the block values for a specific weight layer in the latent vector."""
        layer_blocks = self._to_tensor(layer_blocks)
        z_new = z.clone()
        layer_block_start = self.weight_block_offsets[layer_idx].item()
        layer_block_end = self.weight_block_offsets[layer_idx + 1].item()
        z_new[layer_block_start:layer_block_end] = layer_blocks
        return z_new
    
    def set_blocks_for_biases(self, z: torch.Tensor, 
                             bias_blocks: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Update the block values for all biases in the latent vector."""
        bias_blocks = self._to_tensor(bias_blocks)
        z_new = z.clone()
        bias_block_start = self.bias_block_offset.item()
        z_new[bias_block_start:] = bias_blocks
        return z_new


class LayerwiseHardWeightSharingV3(ParameterSharing):
    """
    Layerwise Hard Weight Sharing V3: Weight layers get d blocks each, each bias parameter is its own block.
    
    This version differs from V2 by giving each bias parameter its own individual block (no sharing among biases),
    while each weight layer still gets d blocks with parameter sharing within the layer.
    
    Structure:
    - Weight layer 1: d blocks (parameters in layer share these d blocks)
    - Weight layer 2: d blocks (parameters in layer share these d blocks)
    - ...
    - Weight layer L: d blocks (parameters in layer share these d blocks)
    - Bias parameters: each bias parameter gets its own block (identity mapping, no sharing)
    
    Total number of blocks = (num_weight_layers * d) + total_bias_size
    Where:
    - num_weight_layers (l) = number of weight layers
    - d = blocks per weight layer
    - total_bias_size (b) = total number of bias parameters
    
    Formula: total_blocks = l * d + b
    
    Args:
        model: PyTorch model
        d: Number of blocks per weight layer
        seed: Random seed for reproducible assignment generation
        alpha: Scaling factor for parameters
        device: Device for computations
        assignment_strategy: How to initialize assignments within each weight layer ('random', 'uniform')
    """
    
    def __init__(self, model: torch.nn.Module, d: int, seed: int = 0,
                 alpha: float = 1.0, device: str = 'cuda', 
                 assignment_strategy: str = 'random', train_biases: bool = True) -> None:

        super().__init__(model, d, alpha, device, seed, train_biases)
        
        self.k = d  # Blocks per weight layer
        self.assignment_strategy = assignment_strategy
        
        # Extract layer information (separating weights and biases)
        self._extract_layer_info()
        
        # Initialize layer-wise assignments
        self.initialize()
        
        print(f"LayerwiseHardWeightSharingV3 initialized:")
        print(f"  - Number of weight layers (l): {self.num_weight_layers}")
        print(f"  - Blocks per weight layer (d): {self.k}")
        print(f"  - Number of bias parameters (b): {self.total_bias_size}")
        print(f"  - Total blocks (l*d + b): {self.d} = {self.num_weight_layers}*{self.k} + {self.total_bias_size}")
        print(f"  - Total parameters (D): {self.D}")
        print(f"  - Compression ratio: {self.get_compression_ratio():.2f}")
        print(self.count_block_sizes())
    
    def _extract_layer_info(self) -> None:
        """Extract layer boundaries, separating weights and biases."""
        self.weight_boundaries = []  # List of (start_idx, end_idx) for weight layers
        self.weight_sizes = []  # Number of parameters in each weight layer
        self.weight_layer_names = []  # Names of weight layers
        
        self.bias_boundaries = []  # List of (start_idx, end_idx) for bias layers
        self.bias_sizes = []  # Number of parameters in each bias layer
        self.bias_layer_names = []  # Names of bias layers
        
        start_idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layer_size = param.numel()
                end_idx = start_idx + layer_size
                
                is_bias = 'bias' in name
                
                if is_bias:
                    self.bias_boundaries.append((start_idx, end_idx))
                    self.bias_sizes.append(layer_size)
                    self.bias_layer_names.append(name)
                else:
                    self.weight_boundaries.append((start_idx, end_idx))
                    self.weight_sizes.append(layer_size)
                    self.weight_layer_names.append(name)
                
                start_idx = end_idx
        
        self.num_weight_layers = len(self.weight_boundaries)
        self.num_bias_params = len(self.bias_boundaries)
        self.total_bias_size = sum(self.bias_sizes)
        
        # Total blocks = (num_weight_layers * d) + total_bias_size
        # Formula: l * d + b
        self.d = self.num_weight_layers * self.k + self.total_bias_size
        
        # Store layer information for easier access
        self.layer_info = {
            'weights': {
                name: {'size': size, 'blocks': self.k, 'is_bias': False}
                for name, size in zip(self.weight_layer_names, self.weight_sizes)
            },
            'biases': {
                name: {'size': size, 'blocks': size, 'is_bias': True}  # Each bias param is its own block
                for name, size in zip(self.bias_layer_names, self.bias_sizes)
            }
        }
        
    def initialize(self) -> None:
        """Initialize parameter assignments to layer-specific blocks."""
        assignments = torch.zeros(self.D, dtype=torch.long, device=self.device)
        
        # Block offset tracking
        # First blocks are for weight layers, then individual blocks for each bias parameter
        current_block_offset = 0
        
        # Assign weight layers (each gets d blocks with sharing)
        for layer_idx, (start, end) in enumerate(self.weight_boundaries):
            layer_size = end - start
            layer_block_start = current_block_offset
            layer_block_end = current_block_offset + self.k
            
            if self.assignment_strategy == 'random':
                # Balanced random assignment: ensure all blocks are used
                layer_assignments = self._balanced_random_assignment(
                    layer_size, layer_block_start, layer_block_end
                )
            elif self.assignment_strategy == 'uniform':
                # Uniform distribution across layer's blocks
                layer_assignments = torch.arange(layer_size, device=self.device) % self.k
                layer_assignments += layer_block_start
            else:
                raise ValueError(f"Unknown assignment strategy: {self.assignment_strategy}")
            
            assignments[start:end] = layer_assignments
            current_block_offset += self.k
        
        # Assign biases: each bias parameter gets its own block (identity mapping)
        # This means bias parameter i maps to block (weight_blocks + i)
        bias_block_start = current_block_offset
        
        for bias_idx, (start, end) in enumerate(self.bias_boundaries):
            bias_size = end - start
            # Identity mapping: each parameter gets its own sequential block
            bias_assignments = torch.arange(
                bias_block_start,
                bias_block_start + bias_size,
                device=self.device
            )
            assignments[start:end] = bias_assignments
            bias_block_start += bias_size
        
        # Store block offsets for weights
        weight_block_offsets = list(range(0, self.num_weight_layers * self.k + 1, self.k))
        bias_block_offset = self.num_weight_layers * self.k
        
        self.register_buffer('weight_block_offsets', 
                            torch.tensor(weight_block_offsets, device=self.device))
        self.register_buffer('bias_block_offset', 
                            torch.tensor(bias_block_offset, device=self.device))
        
        # Store assignments as buffer (non-learnable)
        self.register_buffer('assignments', assignments)
    
    def _balanced_random_assignment(self, num_params: int, block_start: int, block_end: int) -> torch.Tensor:
        """
        Create balanced random assignments that ensure all blocks are used.
        
        Strategy:
        1. First, assign parameters in a round-robin fashion to cover all blocks
        2. Then, randomly shuffle the assignments
        
        This guarantees all blocks get at least one parameter.
        
        Args:
            num_params: Number of parameters to assign
            block_start: Starting block index
            block_end: Ending block index (exclusive)
            
        Returns:
            Tensor of assignments of shape (num_params,)
        """
        num_blocks = block_end - block_start
        
        if num_params < num_blocks:
            # If we have fewer parameters than blocks, just assign sequentially
            assignments = torch.arange(block_start, block_start + num_params, device=self.device)
        else:
            # Create assignments that cycle through all blocks
            # This ensures each block gets at least floor(num_params / num_blocks) parameters
            assignments = torch.arange(num_params, device=self.device) % num_blocks
            assignments += block_start
            
            # Shuffle to make it random while keeping all blocks used
            perm = torch.randperm(num_params, device=self.device)
            assignments = assignments[perm]
        
        return assignments
    
    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector (blocks) to full parameter space.
        
        Args:
            z: Blocks vector of shape (d,) where d = (num_weight_layers * k) + total_bias_size
               The blocks are organized as:
               - First k blocks: weight layer 0 (shared among layer 0 parameters)
               - Next k blocks: weight layer 1 (shared among layer 1 parameters)
               - ...
               - Remaining blocks: individual bias parameters (one-to-one mapping)
            
        Returns:
            Full parameter tensor of shape (D,) where each position gets the value
            from its assigned block: theta[i] = z[assignments[i]]
        """
        z = self._to_tensor(z)
        
        # Validate input dimension
        if z.shape[0] != self.d:
            raise ValueError(f"Expected latent vector of size {self.d}, got {z.shape[0]}")
        
        # Use the input z as the blocks values and look up based on assignments
        x = z[self.assignments]
        # Hard weight sharing already produces full D-dimensional output
        # Just apply alpha scaling directly
        return self.theta_base + self.alpha * x
    
    def count_block_sizes(self) -> dict:
        """Return how many parameters are assigned to each block, organized by layer."""
        block_usage = {}
        
        # Count for weight layers
        for layer_idx, layer_name in enumerate(self.weight_layer_names):
            start, end = self.weight_boundaries[layer_idx]
            layer_block_start = self.weight_block_offsets[layer_idx].item()
            layer_block_end = self.weight_block_offsets[layer_idx + 1].item()
            num_layer_blocks = layer_block_end - layer_block_start
            
            layer_usage = torch.zeros(num_layer_blocks, device=self.device)
            for local_block_idx in range(num_layer_blocks):
                global_block_idx = layer_block_start + local_block_idx
                layer_usage[local_block_idx] = (self.assignments[start:end] == global_block_idx).sum()
            
            block_usage[layer_name] = layer_usage.cpu().numpy()
        
        # Count for bias parameters (each should be exactly 1)
        for bias_idx, bias_name in enumerate(self.bias_layer_names):
            start, end = self.bias_boundaries[bias_idx]
            bias_size = end - start
            
            # For biases, each parameter has its own block, so usage is always 1
            bias_usage = torch.ones(bias_size, device=self.device)
            block_usage[bias_name] = bias_usage.cpu().numpy()
        
        return block_usage
    
    def get_compression_ratio(self) -> float:
        """Return the compression ratio achieved by the blocks."""
        return self.D / self.d
    
    def get_layer_info(self) -> dict:
        """Return detailed information about layer structure."""
        info = {
            'num_weight_layers': self.num_weight_layers,
            'num_bias_params': self.num_bias_params,
            'total_bias_size': self.total_bias_size,
            'blocks_per_weight_layer_k': self.k,
            'total_blocks': self.d,
            'total_parameters': self.D,
            'compression_ratio': self.get_compression_ratio(),
            'weight_layers': [],
            'bias_layers': []
        }
        
        # Weight layers info
        for layer_idx, layer_name in enumerate(self.weight_layer_names):
            start, end = self.weight_boundaries[layer_idx]
            layer_size = end - start
            layer_block_start = self.weight_block_offsets[layer_idx].item()
            layer_block_end = self.weight_block_offsets[layer_idx + 1].item()
            num_layer_blocks = layer_block_end - layer_block_start
            
            layer_info = {
                'name': layer_name,
                'layer_idx': layer_idx,
                'param_range': (start, end),
                'param_size': layer_size,
                'block_range': (layer_block_start, layer_block_end),
                'num_blocks': num_layer_blocks,
                'layer_compression_ratio': layer_size / num_layer_blocks if num_layer_blocks > 0 else 0
            }
            info['weight_layers'].append(layer_info)
        
        # Bias layers info (each parameter is its own block)
        current_bias_block = self.bias_block_offset.item()
        for bias_idx, bias_name in enumerate(self.bias_layer_names):
            start, end = self.bias_boundaries[bias_idx]
            bias_size = end - start
            
            bias_info = {
                'name': bias_name,
                'bias_idx': bias_idx,
                'param_range': (start, end),
                'param_size': bias_size,
                'block_range': (current_bias_block, current_bias_block + bias_size),
                'num_blocks': bias_size,
                'compression_ratio': 1.0  # No compression for biases
            }
            info['bias_layers'].append(bias_info)
            current_bias_block += bias_size
        
        return info
    
    def get_blocks_for_weight_layer(self, layer_idx: int, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract the block values for a specific weight layer from the latent vector."""
        z = self._to_tensor(z)
        layer_block_start = self.weight_block_offsets[layer_idx].item()
        layer_block_end = self.weight_block_offsets[layer_idx + 1].item()
        return z[layer_block_start:layer_block_end]
    
    def get_blocks_for_bias(self, bias_idx: int, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract the block values for a specific bias layer from the latent vector."""
        z = self._to_tensor(z)
        
        # Calculate bias block range
        bias_block_start = self.bias_block_offset.item()
        for i in range(bias_idx):
            bias_block_start += self.bias_sizes[i]
        bias_block_end = bias_block_start + self.bias_sizes[bias_idx]
        
        return z[bias_block_start:bias_block_end]
    
    def get_all_bias_blocks(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract all bias block values from the latent vector."""
        z = self._to_tensor(z)
        bias_block_start = self.bias_block_offset.item()
        return z[bias_block_start:]
    
    def set_blocks_for_weight_layer(self, layer_idx: int, z: torch.Tensor, 
                                   layer_blocks: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Update the block values for a specific weight layer in the latent vector."""
        layer_blocks = self._to_tensor(layer_blocks)
        z_new = z.clone()
        layer_block_start = self.weight_block_offsets[layer_idx].item()
        layer_block_end = self.weight_block_offsets[layer_idx + 1].item()
        z_new[layer_block_start:layer_block_end] = layer_blocks
        return z_new
    
    def set_blocks_for_bias(self, bias_idx: int, z: torch.Tensor, 
                           bias_blocks: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Update the block values for a specific bias layer in the latent vector."""
        bias_blocks = self._to_tensor(bias_blocks)
        z_new = z.clone()
        
        # Calculate bias block range
        bias_block_start = self.bias_block_offset.item()
        for i in range(bias_idx):
            bias_block_start += self.bias_sizes[i]
        bias_block_end = bias_block_start + self.bias_sizes[bias_idx]
        
        z_new[bias_block_start:bias_block_end] = bias_blocks
        return z_new
    
    def set_all_bias_blocks(self, z: torch.Tensor, 
                           bias_blocks: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Update all bias block values in the latent vector."""
        bias_blocks = self._to_tensor(bias_blocks)
        z_new = z.clone()
        bias_block_start = self.bias_block_offset.item()
        z_new[bias_block_start:] = bias_blocks
        return z_new


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
            perm = torch.randperm(self.D, device=self.device)
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
        
        for i, (start, end) in enumerate(self.block_boundaries):
            block_size = end - start
            
            # Determine latent dimension for this block
            if i < self.d_remainder:
                block_d = self.d_per_block + 1
            else:
                block_d = self.d_per_block
            
            # Create projection matrix: block_size x block_d
            P = torch.randn(block_size, block_d, device=self.device)
            
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
        basis = torch.randn(self.D, self.d, device=self.device)
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
        print("Basis reset to random initialize")
    
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
        random_matrix = torch.randn(self.D, self.k, device=self.device)
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
        z_basis = torch.randn(self.d_basis, device=self.device)
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


class LayerwiseRandomProjection(ParameterSharing):
    """
    Layer-wise Random Dense Projection: Each weight layer has its own random projection matrix.
    
    This class implements layer-wise soft parameter sharing by applying separate random projection
    matrices to each weight layer while keeping bias parameters as individual learnable parameters.
    
    Structure:
    - Weight layer 1: theta_1 = theta_base_1 + P_1 @ z_1, where P_1 is a random matrix of shape (D_1, d_1)
    - Weight layer 2: theta_2 = theta_base_2 + P_2 @ z_2, where P_2 is a random matrix of shape (D_2, d_2)
    - ...
    - Weight layer L: theta_L = theta_base_L + P_L @ z_L, where P_L is a random matrix of shape (D_L, d_L)
    - Bias parameters: theta_bias = theta_base_bias + z_bias (identity mapping, no compression)
    
    Total latent dimension d = sum(d_i for all weight layers) + total_bias_size
    
    Args:
        model: PyTorch model
        d: Total latent dimension or latent dimension per weight layer (see d_strategy)
        alpha: Scaling factor for parameters
        normalize: Whether to normalize projection matrices using QR decomposition
        device: Device for computations
        seed: Random seed for reproducible initialization
        d_strategy: How to distribute latent dimensions ('uniform', 'proportional')
                   - 'uniform': Each weight layer gets d latent dimensions
                   - 'proportional': Distribute d proportionally to layer sizes
    """
    
    def __init__(self, model: torch.nn.Module, id: int, alpha: float = 1.0, 
                 normalize: bool = False, device: str = 'cuda', seed: int = 0,
                 d_strategy: str = 'uniform', train_biases: bool = True) -> None:
        
        # Store d_strategy before calling super().__init__
        self.d_strategy = d_strategy
        self.normalize = normalize
        
        # Initialize base class (will set self.d to the input d temporarily)
        super().__init__(model, id, alpha, device, seed, train_biases)
        
        # Extract layer information (separating weights and biases)
        self._extract_layer_info()
        
        # Calculate actual latent dimensions per layer
        self._calculate_latent_dimensions()
        
        # Initialize projection matrices
        self.initialize()
        
        print(f"LayerwiseRandomProjection initialized:")
        print(f"  - Number of weight layers (l): {self.num_weight_layers}")
        print(f"  - Total bias parameters (b): {self.total_bias_size}")
        print(f"  - Total latent dimension (d): {self.d} = sum(d_i) + {self.total_bias_size}")
        print(f"  - Total parameters (D): {self.D}")
        print(f"  - Compression ratio: {self.get_compression_ratio():.2f}")
        print(f"  - d_strategy: {self.d_strategy}")
        print(f"  - normalize: {self.normalize}")
    
    def _extract_layer_info(self) -> None:
        """Extract layer boundaries, separating weights and biases."""
        self.weight_boundaries = []  # List of (start_idx, end_idx) for weight layers
        self.weight_sizes = []  # Number of parameters in each weight layer
        self.weight_layer_names = []  # Names of weight layers
        
        self.bias_boundaries = []  # List of (start_idx, end_idx) for bias layers
        self.bias_sizes = []  # Number of parameters in each bias layer
        self.bias_layer_names = []  # Names of bias layers
        
        start_idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layer_size = param.numel()
                end_idx = start_idx + layer_size
                
                is_bias = 'bias' in name
                
                if is_bias:
                    self.bias_boundaries.append((start_idx, end_idx))
                    self.bias_sizes.append(layer_size)
                    self.bias_layer_names.append(name)
                else:
                    self.weight_boundaries.append((start_idx, end_idx))
                    self.weight_sizes.append(layer_size)
                    self.weight_layer_names.append(name)
                
                start_idx = end_idx
        
        self.num_weight_layers = len(self.weight_boundaries)
        self.num_bias_params = len(self.bias_boundaries)
        self.total_bias_size = sum(self.bias_sizes)
    
    def _calculate_latent_dimensions(self) -> None:
        """Calculate latent dimensions for each weight layer based on strategy."""
        if self.d_strategy == 'uniform':
            # Each weight layer gets the same latent dimension d
            # Total latent dimension = num_weight_layers * d + (total_bias_size if train_biases else 0)
            d_per_layer = self.id
            self.weight_latent_dims = [d_per_layer] * self.num_weight_layers
            
        elif self.d_strategy == 'proportional':
            # Distribute d proportionally to layer sizes
            # d_i = floor((D_i / sum(D_j)) * d_total_for_weights)
            # where d_total_for_weights = d - (total_bias_size if train_biases else 0)
            
            total_weight_params = sum(self.weight_sizes)
            bias_contribution = self.total_bias_size if self.train_biases else 0
            d_total_for_weights = max(1, self.id - bias_contribution)
            
            self.weight_latent_dims = []
            allocated_d = 0
            
            for i, layer_size in enumerate(self.weight_sizes):
                if i == self.num_weight_layers - 1:
                    # Last layer gets the remainder
                    d_i = d_total_for_weights - allocated_d
                else:
                    # Proportional allocation
                    proportion = layer_size / total_weight_params
                    d_i = max(1, int(proportion * d_total_for_weights))
                    allocated_d += d_i
                
                self.weight_latent_dims.append(d_i)
        else:
            raise ValueError(f"Unknown d_strategy: {self.d_strategy}")
        
        # Calculate total latent dimension
        # If train_biases=False, biases are not part of latent space
        bias_contribution = self.total_bias_size if self.train_biases else 0
        self.d = sum(self.weight_latent_dims) + bias_contribution
        
        # Create latent dimension boundaries for easy indexing
        self.weight_latent_offsets = [0]
        cumsum = 0
        for d_i in self.weight_latent_dims:
            cumsum += d_i
            self.weight_latent_offsets.append(cumsum)
        
        self.bias_latent_offset = cumsum
    
    def _get_projection_buffer_name(self, layer_name: str) -> str:
        """Get the buffer name for a layer's projection matrix."""
        # Replace dots and other special characters with underscores
        sanitized_name = layer_name.replace(".", "_")
        return f'P_{sanitized_name}'
    
    def initialize(self) -> None:
        """Initialize random projection matrices for each weight layer."""
        # Create projection matrices for each weight layer
        self.projection_matrices = nn.ModuleList()
        
        for layer_idx in range(self.num_weight_layers):
            layer_size = self.weight_sizes[layer_idx]
            d_i = self.weight_latent_dims[layer_idx]
            layer_name = self.weight_layer_names[layer_idx]
            
            # Create random projection matrix: layer_size x d_i
            P_i = torch.randn(layer_size, d_i, device=self.device)
            
            if self.normalize:
                # Apply QR decomposition for orthonormal columns
                # P_i, _ = torch.linalg.qr(P_i)
                P_i = P_i / P_i.norm(dim=0, keepdim=True)
                # Scale by 1/sqrt(d_i)
                P_i = P_i / (d_i ** 0.5)
            
            # Store as buffer (non-learnable) using sanitized layer name
            buffer_name = self._get_projection_buffer_name(layer_name)
            self.register_buffer(buffer_name, P_i)
    
    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using layer-wise random projections.
        
        Args:
            z: Latent vector of shape (d,) where d = sum(d_i for weight layers) + total_bias_size
               The latent vector is organized as:
               - z[0:d_0]: latent for weight layer 0
               - z[d_0:d_0+d_1]: latent for weight layer 1
               - ...
               - z[sum(d_i):]: latent for bias parameters (identity mapping)
            
        Returns:
            Full parameter tensor of shape (D,)
        """
        z = self._to_tensor(z)
        
        # Validate input dimension
        if z.shape[0] != self.d:
            raise ValueError(f"Expected latent vector of size {self.d}, got {z.shape[0]}")
        
        # Initialize output tensor
        theta = torch.zeros(self.D, device=self.device)
        
        # Apply layer-wise projections for weight layers
        for layer_idx in range(self.num_weight_layers):
            start_param, end_param = self.weight_boundaries[layer_idx]
            start_latent = self.weight_latent_offsets[layer_idx]
            end_latent = self.weight_latent_offsets[layer_idx + 1]
            layer_name = self.weight_layer_names[layer_idx]
            
            # Extract latent for this layer
            z_i = z[start_latent:end_latent]
            
            # Apply projection: theta_i = P_i @ z_i
            buffer_name = self._get_projection_buffer_name(layer_name)
            P_i = getattr(self, buffer_name)
            theta_i = P_i @ z_i
            
            # Store in output tensor
            theta[start_param:end_param] = theta_i
        
        # Handle bias parameters (identity mapping) - only if train_biases=True
        if self.train_biases:
            for bias_idx in range(self.num_bias_params):
                start_param, end_param = self.bias_boundaries[bias_idx]
                bias_size = end_param - start_param
                
                # Extract bias latent (one-to-one mapping)
                bias_latent_start = self.bias_latent_offset + sum(self.bias_sizes[:bias_idx])
                bias_latent_end = bias_latent_start + bias_size
                
                z_bias = z[bias_latent_start:bias_latent_end]
                theta[start_param:end_param] = z_bias
        # else: biases remain zero in theta, so theta_base values are preserved
        
        # Apply scaling and add to base parameters
        theta = self.process(theta)
        return self.theta_base + theta
    
    def get_compression_ratio(self) -> float:
        """Return the compression ratio achieved by the projection."""
        return self.D / self.d
    
    def get_layer_info(self) -> dict:
        """Return detailed information about layer structure."""
        info = {
            'num_weight_layers': self.num_weight_layers,
            'num_bias_params': self.num_bias_params,
            'total_bias_size': self.total_bias_size,
            'total_latent_dim': self.d,
            'total_parameters': self.D,
            'compression_ratio': self.get_compression_ratio(),
            'weight_layers': [],
            'bias_layers': [],
            'd_strategy': self.d_strategy,
            'normalize': self.normalize
        }
        
        # Weight layers info
        for layer_idx in range(self.num_weight_layers):
            start_param, end_param = self.weight_boundaries[layer_idx]
            start_latent = self.weight_latent_offsets[layer_idx]
            end_latent = self.weight_latent_offsets[layer_idx + 1]
            
            layer_info = {
                'name': self.weight_layer_names[layer_idx],
                'layer_idx': layer_idx,
                'param_range': (start_param, end_param),
                'param_size': self.weight_sizes[layer_idx],
                'latent_range': (start_latent, end_latent),
                'latent_dim': self.weight_latent_dims[layer_idx],
                'layer_compression_ratio': self.weight_sizes[layer_idx] / self.weight_latent_dims[layer_idx]
            }
            info['weight_layers'].append(layer_info)
        
        # Bias layers info (each parameter is its own latent dimension)
        for bias_idx in range(self.num_bias_params):
            start_param, end_param = self.bias_boundaries[bias_idx]
            bias_size = self.bias_sizes[bias_idx]
            bias_latent_start = self.bias_latent_offset + sum(self.bias_sizes[:bias_idx])
            bias_latent_end = bias_latent_start + bias_size
            
            bias_info = {
                'name': self.bias_layer_names[bias_idx],
                'bias_idx': bias_idx,
                'param_range': (start_param, end_param),
                'param_size': bias_size,
                'latent_range': (bias_latent_start, bias_latent_end),
                'latent_dim': bias_size,
                'compression_ratio': 1.0  # No compression for biases
            }
            info['bias_layers'].append(bias_info)
        
        return info
    
    def get_projection_matrix(self, layer_idx: int) -> torch.Tensor:
        """Get the projection matrix for a specific weight layer."""
        if layer_idx >= self.num_weight_layers:
            raise ValueError(f"Invalid layer_idx: {layer_idx}. Only {self.num_weight_layers} weight layers exist.")
        layer_name = self.weight_layer_names[layer_idx]
        buffer_name = self._get_projection_buffer_name(layer_name)
        return getattr(self, buffer_name)
    
    def get_latent_for_weight_layer(self, layer_idx: int, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract the latent vector for a specific weight layer from the full latent vector."""
        z = self._to_tensor(z)
        start_latent = self.weight_latent_offsets[layer_idx]
        end_latent = self.weight_latent_offsets[layer_idx + 1]
        return z[start_latent:end_latent]
    
    def get_latent_for_bias(self, bias_idx: int, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract the latent vector for a specific bias layer from the full latent vector."""
        z = self._to_tensor(z)
        bias_latent_start = self.bias_latent_offset + sum(self.bias_sizes[:bias_idx])
        bias_latent_end = bias_latent_start + self.bias_sizes[bias_idx]
        return z[bias_latent_start:bias_latent_end]
    
    def set_latent_for_weight_layer(self, layer_idx: int, z: torch.Tensor, 
                                    layer_latent: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Update the latent values for a specific weight layer in the full latent vector."""
        layer_latent = self._to_tensor(layer_latent)
        z_new = z.clone()
        start_latent = self.weight_latent_offsets[layer_idx]
        end_latent = self.weight_latent_offsets[layer_idx + 1]
        z_new[start_latent:end_latent] = layer_latent
        return z_new
    
    def set_latent_for_bias(self, bias_idx: int, z: torch.Tensor, 
                           bias_latent: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Update the latent values for a specific bias layer in the full latent vector."""
        bias_latent = self._to_tensor(bias_latent)
        z_new = z.clone()
        bias_latent_start = self.bias_latent_offset + sum(self.bias_sizes[:bias_idx])
        bias_latent_end = bias_latent_start + self.bias_sizes[bias_idx]
        z_new[bias_latent_start:bias_latent_end] = bias_latent
        return z_new


class GlobalRandomProjection(ParameterSharing):
    """
    Global Random Projection: Single projection matrix for all weight parameters.

    This class implements layer-wise soft parameter sharing by applying a single random projection
    matrix to all weight parameters collectively, while keeping bias parameters as individual
    learnable parameters with identity mapping.

    Structure:
    - Weight parameters: theta_weights = P @ z_weights, where P is a random matrix of shape (total_weight_params, d)
    - Bias parameters: theta_bias = z_bias (identity mapping, no compression)
    - Total latent dimension d_total = d + total_bias_size

    Args:
        model: PyTorch model
        d: Latent dimension for weight parameters
        alpha: Scaling factor for parameters
        normalize: Whether to normalize projection matrices using QR decomposition
        device: Device for computations
        seed: Random seed for reproducible initialization
    """

    def __init__(self, model: torch.nn.Module, d: int, alpha: float = 1.0,
                 normalize: bool = False, device: str = 'cuda', seed: int = 0, train_biases: bool = True) -> None:

        # Store parameters before calling super().__init__
        self.normalize = normalize
        self._weight_latent_dim = d  # Store the input d for weight latent dimension

        # Initialize base class (will set self.d to the input d temporarily)
        super().__init__(model, d, alpha, device, seed, train_biases)

        # Extract layer information (separating weights and biases)
        self._extract_layer_info()

        # Calculate actual latent dimension: d (for weights) + (total_bias_size if train_biases else 0)
        bias_contribution = self.total_bias_size if self.train_biases else 0
        self.d = self._weight_latent_dim + bias_contribution

        # Initialize projection matrix for weights
        self.initialize()

        print(f"GlobalRandomProjection initialized:")
        print(f"  - Total weight parameters (W): {self.total_weight_size}")
        print(f"  - Total bias parameters (b): {self.total_bias_size}")
        print(f"  - Weight latent dimension (d): {self._weight_latent_dim}")
        print(f"  - Total latent dimension: {self.d} = {self._weight_latent_dim} + {self.total_bias_size}")
        print(f"  - Total parameters (D): {self.D}")
        print(f"  - Weight compression ratio: {self.total_weight_size / self._weight_latent_dim:.2f}")
        print(f"  - normalize: {self.normalize}")

    def _extract_layer_info(self) -> None:
        """Extract layer boundaries, separating weights and biases."""
        self.weight_boundaries = []  # List of (start_idx, end_idx) for weight layers
        self.weight_sizes = []  # Number of parameters in each weight layer
        self.weight_layer_names = []  # Names of weight layers

        self.bias_boundaries = []  # List of (start_idx, end_idx) for bias layers
        self.bias_sizes = []  # Number of parameters in each bias layer
        self.bias_layer_names = []  # Names of bias layers

        start_idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layer_size = param.numel()
                end_idx = start_idx + layer_size

                is_bias = 'bias' in name

                if is_bias:
                    self.bias_boundaries.append((start_idx, end_idx))
                    self.bias_sizes.append(layer_size)
                    self.bias_layer_names.append(name)
                else:
                    self.weight_boundaries.append((start_idx, end_idx))
                    self.weight_sizes.append(layer_size)
                    self.weight_layer_names.append(name)

                start_idx = end_idx

        self.num_weight_layers = len(self.weight_boundaries)
        self.num_bias_params = len(self.bias_boundaries)
        self.total_weight_size = sum(self.weight_sizes)
        self.total_bias_size = sum(self.bias_sizes)

        # Create weight parameter offsets for easy indexing
        self.weight_offsets = [0]
        cumsum = 0
        for size in self.weight_sizes:
            cumsum += size
            self.weight_offsets.append(cumsum)

    def initialize(self) -> None:
        """Initialize the random projection matrix for all weight parameters."""
        # Create single projection matrix for all weights: total_weight_size x d
        P = torch.randn(self.total_weight_size, self._weight_latent_dim, device=self.device)

        if self.normalize:
            # Apply normalization: each column has unit norm, then scale by 1/sqrt(d)
            # P = P / P.norm(dim=0, keepdim=True)
            P, _ = torch.linalg.qr(P)
            P = P / (self._weight_latent_dim ** 0.5)

        # Store as buffer (non-learnable)
        self.register_buffer('P', P)

    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using global random projection.

        Args:
            z: Latent vector of shape (d_total,) where d_total = d + total_bias_size
               The latent vector is organized as:
               - z[0:d]: latent for all weight parameters (global projection)
               - z[d:]: latent for bias parameters (identity mapping)

        Returns:
            Full parameter tensor of shape (D,)
        """
        z = self._to_tensor(z)

        # Validate input dimension
        if z.shape[0] != self.d:
            raise ValueError(f"Expected latent vector of size {self.d}, got {z.shape[0]}")

        # Split latent vector
        z_weights = z[:self._weight_latent_dim]  # Shape: (d,)
        
        # Initialize output tensor
        theta = torch.zeros(self.D, device=self.device)

        # Apply global projection for all weight parameters
        theta_weights = self.P @ z_weights  # Shape: (total_weight_size,)

        # Place weight parameters in their correct positions
        weight_idx = 0
        for layer_idx in range(self.num_weight_layers):
            start_param, end_param = self.weight_boundaries[layer_idx]
            layer_size = self.weight_sizes[layer_idx]

            theta[start_param:end_param] = theta_weights[weight_idx:weight_idx + layer_size]
            weight_idx += layer_size

        # Handle bias parameters (identity mapping) - only if train_biases=True
        if self.train_biases:
            z_biases = z[self._weight_latent_dim:]   # Shape: (total_bias_size,)
            bias_offset = 0
            for bias_idx in range(self.num_bias_params):
                start_param, end_param = self.bias_boundaries[bias_idx]
                bias_size = self.bias_sizes[bias_idx]

                theta[start_param:end_param] = z_biases[bias_offset:bias_offset + bias_size]
                bias_offset += bias_size
        # else: biases remain zero in theta, so theta_base values are preserved

        # Apply scaling and add to base parameters
        theta = self.process(theta)
        return self.theta_base + theta

    def get_compression_ratio(self) -> float:
        """Return the compression ratio achieved by the projection (for weights only)."""
        return self.total_weight_size / self._weight_latent_dim

    def get_layer_info(self) -> dict:
        """Return detailed information about layer structure."""
        info = {
            'num_weight_layers': self.num_weight_layers,
            'num_bias_params': self.num_bias_params,
            'total_weight_size': self.total_weight_size,
            'total_bias_size': self.total_bias_size,
            'weight_latent_dim': self._weight_latent_dim,
            'total_latent_dim': self.d,
            'total_parameters': self.D,
            'weight_compression_ratio': self.get_compression_ratio(),
            'weight_layers': [],
            'bias_layers': [],
            'normalize': self.normalize
        }

        # Weight layers info
        for layer_idx in range(self.num_weight_layers):
            start_param, end_param = self.weight_boundaries[layer_idx]
            start_weight = self.weight_offsets[layer_idx]
            end_weight = self.weight_offsets[layer_idx + 1]

            layer_info = {
                'name': self.weight_layer_names[layer_idx],
                'layer_idx': layer_idx,
                'param_range': (start_param, end_param),
                'param_size': self.weight_sizes[layer_idx],
                'weight_range': (start_weight, end_weight),
            }
            info['weight_layers'].append(layer_info)

        # Bias layers info (each parameter is its own latent dimension)
        bias_offset = 0
        for bias_idx in range(self.num_bias_params):
            start_param, end_param = self.bias_boundaries[bias_idx]
            bias_size = self.bias_sizes[bias_idx]
            bias_latent_start = self._weight_latent_dim + bias_offset
            bias_latent_end = bias_latent_start + bias_size

            bias_info = {
                'name': self.bias_layer_names[bias_idx],
                'bias_idx': bias_idx,
                'param_range': (start_param, end_param),
                'param_size': bias_size,
                'latent_range': (bias_latent_start, bias_latent_end),
                'compression_ratio': 1.0  # No compression for biases
            }
            info['bias_layers'].append(bias_info)
            bias_offset += bias_size

        return info

    def get_projection_matrix(self) -> torch.Tensor:
        """Get the global projection matrix."""
        return self.P

    def get_latent_for_weights(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract the latent vector for weight parameters from the full latent vector."""
        z = self._to_tensor(z)
        return z[:self._weight_latent_dim]

    def get_latent_for_biases(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Extract the latent vector for bias parameters from the full latent vector."""
        z = self._to_tensor(z)
        return z[self._weight_latent_dim:]

    def set_latent_for_weights(self, z: torch.Tensor,
                              weight_latent: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Update the latent values for weight parameters in the full latent vector."""
        weight_latent = self._to_tensor(weight_latent)
        z_new = z.clone()
        z_new[:self._weight_latent_dim] = weight_latent
        return z_new

    def set_latent_for_biases(self, z: torch.Tensor,
                             bias_latent: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Update the latent values for bias parameters in the full latent vector."""
        bias_latent = self._to_tensor(bias_latent)
        z_new = z.clone()
        z_new[self._weight_latent_dim:] = bias_latent
        return z_new


class LoRAParameterSharing(ParameterSharing):
    """
    Low-Rank Adaptation (LoRA) parameter sharing.

    This class implements parameter sharing using LoRA (Low-Rank Adaptation) where each
    weight layer is adapted using low-rank matrices A and B.

    For each weight layer W (shape: out_features x in_features):
    - A: matrix of shape (rank x in_features)
    - B: matrix of shape (out_features x rank)
    - Adaptation: ΔW = B @ A
    - Final weight: W = W_base + ΔW

    The latent vector z consists of:
    - LoRA parameters for all weight layers (flattened A and B matrices)
    - Bias parameters for all bias layers (optional, if train_biases=True)

    Bias handling:
    - If train_biases=True: Biases are included in latent space (identity mapping)
    - If train_biases=False: Biases are frozen at their base values

    Args:
        model: PyTorch model
        d: LoRA rank for weight adaptation
        alpha: Scaling factor for parameters
        device: Device for computations
        seed: Random seed for initialization
        train_biases: Whether to include biases in the latent space (default: False)
    """

    def __init__(self, model: torch.nn.Module, rank: int, alpha: float = 1.0,
                 device: str = 'cuda', seed: int = 0, train_biases: bool = False) -> None:

        super().__init__(model, rank, alpha, device, seed)

        self.rank = rank  # LoRA rank
        self.train_biases = train_biases  # Whether to include biases in latent space
        self._extract_layer_info()
        self.initialize()

        print(f"LoRA Parameter Sharing initialized:")
        print(f"  - LoRA rank: {self.rank}")
        print(f"  - Train biases: {self.train_biases}")
        print(f"  - Number of weight layers: {self.num_weight_layers}")
        print(f"  - Number of bias layers: {self.num_bias_params} ({'trainable' if self.train_biases else 'frozen'})")
        print(f"  - Weight latent dimension: {self.weight_latent_dim}")
        if self.train_biases:
            print(f"  - Bias latent dimension: {self.total_bias_size}")
        print(f"  - Total latent dimension (d): {self.d}")
        print(f"  - Total parameters (D): {self.D}")
        print(f"  - Compression ratio: {self.get_compression_ratio():.2f}")
        print(f"  - Device: {self.device} (theta_base on: {self.theta_base.device})")

    def _extract_layer_info(self) -> None:
        """Extract layer boundaries, separating weights and biases."""
        self.weight_boundaries = []  # List of (start_idx, end_idx) for weight layers
        self.weight_sizes = []  # Number of parameters in each weight layer
        self.weight_layer_names = []  # Names of weight layers
        self.weight_shapes = []  # Shapes of weight matrices

        self.bias_boundaries = []  # List of (start_idx, end_idx) for bias layers
        self.bias_sizes = []  # Number of parameters in each bias layer
        self.bias_layer_names = []  # Names of bias layers

        start_idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layer_size = param.numel()
                end_idx = start_idx + layer_size

                is_bias = 'bias' in name

                if is_bias:
                    self.bias_boundaries.append((start_idx, end_idx))
                    self.bias_sizes.append(layer_size)
                    self.bias_layer_names.append(name)
                else:
                    self.weight_boundaries.append((start_idx, end_idx))
                    self.weight_sizes.append(layer_size)
                    self.weight_layer_names.append(name)
                    self.weight_shapes.append(param.shape)

                start_idx = end_idx

        self.num_weight_layers = len(self.weight_boundaries)
        self.num_bias_params = len(self.bias_boundaries)
        self.total_bias_size = sum(self.bias_sizes)

        # Calculate LoRA parameter sizes for each weight layer
        self.lora_param_sizes = []
        self.lora_param_offsets = [0]

        for shape in self.weight_shapes:
            if len(shape) == 2:  # Linear layer: (out_features, in_features)
                out_features, in_features = shape
                # LoRA parameters: A (rank x in_features) + B (out_features x rank)
                lora_size = self.rank * in_features + out_features * self.rank
            elif len(shape) == 4:  # Conv layer: (out_channels, in_channels, kernel_h, kernel_w)
                out_channels, in_channels, kh, kw = shape
                # For conv layers, treat as linear: in_features = in_channels * kh * kw
                in_features = in_channels * kh * kw
                lora_size = self.rank * in_features + out_channels * self.rank
            else:
                # Fallback: treat as 1D parameter vector
                lora_size = min(shape.numel(), self.rank * 2)  # Conservative estimate

            self.lora_param_sizes.append(lora_size)
            self.lora_param_offsets.append(self.lora_param_offsets[-1] + lora_size)

        # Total latent dimension = sum of LoRA params for weights + (optionally) bias params
        self.weight_latent_dim = sum(self.lora_param_sizes)
        if self.train_biases:
            self.d = self.weight_latent_dim + self.total_bias_size
        else:
            self.d = self.weight_latent_dim

    def initialize(self) -> None:
        """Initialize LoRA parameters and pre-allocate buffers."""
        # Ensure theta_base is on the correct device
        self.theta_base = self.theta_base.to(self.device)
        
        # Pre-allocate buffer for reconstructed parameters (reused in forward)
        self.register_buffer('_reconstructed_params', torch.zeros_like(self.theta_base))
        
        # Pre-compute slices for faster access
        self._prepare_slices()

    def _prepare_slices(self) -> None:
        """Pre-compute slice information to avoid repeated computation."""
        # Pre-compute A and B sizes for each layer
        self.a_sizes = []
        self.b_sizes = []
        
        for shape in self.weight_shapes:
            if len(shape) == 2:  # Linear layer
                out_features, in_features = shape
                a_size = self.rank * in_features
                b_size = out_features * self.rank
            elif len(shape) == 4:  # Conv layer
                out_channels, in_channels, kh, kw = shape
                in_features = in_channels * kh * kw
                a_size = self.rank * in_features
                b_size = out_channels * self.rank
            else:
                a_size = 0
                b_size = 0
            
            self.a_sizes.append(a_size)
            self.b_sizes.append(b_size)

    def get_compression_ratio(self) -> float:
        """Calculate compression ratio (D/d)."""
        return self.D / self.d

    def forward(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using LoRA (optimized version).

        Args:
            z: Latent vector of shape (d,) containing LoRA params + (optionally) bias params

        Returns:
            Full parameter tensor of shape (D,)
        """
        z = self._to_tensor(z)

        # Split latent vector into weight LoRA params and bias params
        weight_latent = z[:self.weight_latent_dim]
        if self.train_biases:
            bias_latent = z[self.weight_latent_dim:]

        # Start with base parameters (in-place copy to pre-allocated buffer)
        self._reconstructed_params.copy_(self.theta_base)

        # Process weight layers with LoRA
        weight_latent_idx = 0
        for layer_idx, (start, end) in enumerate(self.weight_boundaries):
            shape = self.weight_shapes[layer_idx]
            
            # Extract LoRA parameters for this layer
            lora_size = self.lora_param_sizes[layer_idx]
            layer_lora_params = weight_latent[weight_latent_idx:weight_latent_idx + lora_size]
            weight_latent_idx += lora_size

            # Reconstruct weight with LoRA adaptation
            if len(shape) == 2:  # Linear layer
                out_features, in_features = shape
                # Use pre-computed sizes
                a_size = self.a_sizes[layer_idx]
                
                # Reshape in one go - avoid intermediate variables
                A = layer_lora_params[:a_size].view(self.rank, in_features)
                B = layer_lora_params[a_size:].view(out_features, self.rank)

                # ΔW = B @ A, flatten, scale, and add in-place (use mm for better GPU performance)
                delta_W_flat = torch.mm(B, A).view(-1)
                self._reconstructed_params[start:end].add_(delta_W_flat, alpha=self.alpha)

            elif len(shape) == 4:  # Conv layer
                out_channels, in_channels, kh, kw = shape
                in_features = in_channels * kh * kw
                
                # Use pre-computed sizes
                a_size = self.a_sizes[layer_idx]
                
                # Reshape A and B
                A = layer_lora_params[:a_size].view(self.rank, in_features)
                B = layer_lora_params[a_size:].view(out_channels, self.rank)

                # ΔW = B @ A, flatten directly (use mm for better GPU performance)
                delta_W_flat = torch.mm(B, A).view(-1)
                self._reconstructed_params[start:end].add_(delta_W_flat, alpha=self.alpha)

        # Process bias layers (direct mapping with in-place operations)
        # Only update biases if train_biases=True, otherwise keep base values
        if self.train_biases:
            bias_latent_idx = 0
            for start, end in self.bias_boundaries:
                bias_size = end - start
                # Directly copy and scale biases in-place
                self._reconstructed_params[start:end].add_(
                    bias_latent[bias_latent_idx:bias_latent_idx + bias_size], 
                    alpha=self.alpha
                )
                bias_latent_idx += bias_size

        return self._reconstructed_params

    def get_layer_info(self) -> dict:
        """Get information about layer organization."""
        info = {
            'num_weight_layers': self.num_weight_layers,
            'num_bias_layers': self.num_bias_params,
            'lora_rank': self.rank,
            'weight_latent_dim': self.weight_latent_dim,
            'bias_latent_dim': self.total_bias_size,
            'total_latent_dim': self.d,
            'compression_ratio': self.get_compression_ratio(),
            'weight_layers': [],
            'bias_layers': []
        }

        # Weight layer info
        for layer_idx, layer_name in enumerate(self.weight_layer_names):
            start, end = self.weight_boundaries[layer_idx]
            layer_info = {
                'name': layer_name,
                'shape': self.weight_shapes[layer_idx],
                'size': self.weight_sizes[layer_idx],
                'lora_params': self.lora_param_sizes[layer_idx],
                'param_range': (start, end)
            }
            info['weight_layers'].append(layer_info)

        # Bias layer info
        for bias_idx, bias_name in enumerate(self.bias_layer_names):
            start, end = self.bias_boundaries[bias_idx]
            bias_info = {
                'name': bias_name,
                'size': self.bias_sizes[bias_idx],
                'param_range': (start, end)
            }
            info['bias_layers'].append(bias_info)

        return info


def create_weight_sharing(model, args):
    """
    Create a weight sharing instance based on the provided arguments.
    
    Args:
        model: PyTorch model
        args: Arguments object containing weight sharing configuration
        optimizer_type: Optional optimizer type (for strategy-specific logic)
        
    Returns:
        Weight sharing instance
    """
    
    param_sharing_type = args.ws.lower()
    seed = args.seed if args.seed is not None else 0
    train_biases = getattr(args, 'train_biases', False)  # Default to False if not specified
    if train_biases:
        print("Training biases in optimization dimensions")
    else:
        print("Freezing biases")
    
    if param_sharing_type == 'block':
        ws = HardWeightSharing(
            model=model, 
            id=args.id, 
            seed=seed, 
            device=args.ws_device,
            train_biases=train_biases
        )
    elif param_sharing_type == 'lwb-hard':
        ws = LayerwiseHardWeightSharing(
            model=model, 
            id=args.id, 
            seed=seed, 
            device=args.ws_device,
            train_biases=train_biases
        )
    elif param_sharing_type == 'lwb-v2':
        ws = LayerwiseHardWeightSharingV2(
            model=model, 
            id=args.id, 
            seed=seed, 
            device=args.ws_device,
            train_biases=train_biases
        )
    elif param_sharing_type == 'lwb-v3':
        ws = LayerwiseHardWeightSharingV3(
            model=model, 
            id=args.id, 
            seed=seed, 
            device=args.ws_device,
            train_biases=train_biases
        )
    elif param_sharing_type == 'dense':
        normalize = args.normalize_projection
        ws = RandomProjectionSoftSharing(
            model=model, 
            id=args.id, 
            alpha=args.alpha, 
            normalize=normalize, 
            seed=seed,
            device=args.ws_device,
            train_biases=train_biases
        )
    elif param_sharing_type == 'lwdp': # layer-wise random projection with proportional distribution
        normalize = getattr(args, 'normalize_projection', False)
        d_strategy = getattr(args, 'd_strategy', 'uniform')
        ws = LayerwiseRandomProjection(
            model=model,
            id=args.id,
            alpha=args.alpha,
            normalize=normalize,
            d_strategy=d_strategy,
            seed=seed,
            device=args.ws_device,
            train_biases=train_biases
        )
    elif param_sharing_type == 'dp2': # global random projection (single P for all weights)
        normalize = getattr(args, 'normalize_projection', False)
        ws = GlobalRandomProjection(
            model=model,
            id=args.id,
            alpha=args.alpha,
            normalize=normalize,
            seed=seed,
            device=args.ws_device,
            train_biases=train_biases
        )
    elif param_sharing_type == 'sparse':
        ws = SparseProjection(
            model=model, 
            id=args.id, 
            alpha=args.alpha, 
            seed=seed,
            device=args.ws_device,
            train_biases=train_biases
        )
    elif param_sharing_type == 'fastfood':
        ws = FastfoodProjection(
            model=model,
            d=args.id,
            alpha=args.alpha,
            seed=seed,
            device=args.ws_device,
            train_biases=train_biases
        )
    elif param_sharing_type == 'mlp':
        hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
        activation = args.activation.lower() if args.activation else None
        ws = MLPSoftSharing(
            model=model, 
            d=args.id, 
            hidden_dims=hidden_dims, 
            use_activation=args.use_activation, 
            activation=activation, 
            alpha=args.alpha, 
            seed=seed,
            device=args.ws_device,
            train_biases=train_biases
        )
    elif param_sharing_type == 'hypernetwork':
        ws = HyperNetworkSoftSharing(
            model=model, 
            id=args.id, 
            alpha=args.alpha, 
            seed=seed,
            device=args.ws_device,
            train_biases=train_biases
        )
    elif param_sharing_type == 'rff':
        ws = RandomFourierFeaturesSoftSharing(
            model=model,
            id=args.id,
            alpha=args.alpha,
            seed=seed,
            device=args.ws_device,
            train_biases=train_biases
        )
    elif param_sharing_type == 'lora':
        train_biases = getattr(args, 'train_biases', False)  # Default to False if not specified
        ws = LoRAParameterSharing(
            model=model,
            rank=args.lora_rank,
            alpha=args.alpha,
            seed=seed,
            device=args.ws_device,
            train_biases=train_biases
        )
    else:
        raise ValueError(f"Unknown weight sharing type: {param_sharing_type}")
    
    return ws