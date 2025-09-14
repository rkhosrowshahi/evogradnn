import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.fft import irfft, fft
import torch
import torch.nn as nn
from typing import Union, List, Optional

from .utils import params_to_vector



class ParameterSharing:
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
    
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, d: int, device: str = 'cuda') -> None:
        """
        Initialize parameter sharing.
        
        Args:
            model: PyTorch model (e.g., ResNet-18).
            criterion: Loss function for the model.
            d: Number of shared parameters (latent dimension K).
            device: Device for PyTorch computations.
        """
        self.device = device
        self.model = model
        self.theta_0 = params_to_vector(self.model.parameters())
        self.D = len(self.theta_0)
        self.d = d
        self.criterion = criterion

        self.assignments = np.random.randint(0, self.d, (self.D,))

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
        self.theta_0 = params_to_vector(self.model.parameters())
        self.D = len(self.theta_0)

    def set_theta(self, theta: Union[np.ndarray, torch.Tensor]) -> None:
        """Set the base parameter vector.
        
        Args:
            theta: Base parameter vector of shape (D,)
        """
        self.theta_0 = theta

    def expand(self, z: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Map latent vector to full parameter space using simple assignment strategy.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Expanded parameter vector of shape (D,)
        """
        x = z[self.assignments].copy()
        return x
    
    def process(self, x: Union[np.ndarray, torch.Tensor], alpha: float = 1.0, max_norm: float = 1.0) -> torch.Tensor:
        """
        Process expanded parameters with scaling and optional normalization.
        
        Args:
            x: Expanded parameter vector of shape (D,)
            alpha: Scaling factor for the parameters
            max_norm: Maximum norm for parameter clipping (currently disabled)
            
        Returns:
            Processed parameter tensor of shape (D,)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
            
        x = x.to(self.device)

        # x_norm = torch.norm(x)
        # if x_norm > max_norm:
        #     x = x / x_norm
        
        theta = alpha * x
        
        return theta
    
    def load_to_model(self, theta: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Load processed parameters into the neural network model.
        
        Args:
            theta: Parameter tensor of shape (D,) to load into model
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.from_numpy(theta).to(self.device).float()
            
        theta = theta.to(self.device)
        
        torch.nn.utils.vector_to_parameters(theta, self.model.parameters())


class GaussianRBFParameterSharing(ParameterSharing):
    """
    Parameter sharing using Gaussian Radial Basis Functions (RBF).
    
    This class maps from a low-dimensional latent space to the full parameter space
    using a weighted sum of Gaussian RBF kernels. Each kernel is parameterized by
    an amplitude, center location, and width (gamma).
    
    The latent vector z contains:
    - Amplitudes: a_i ~ N(0, 0.5) for each RBF
    - Centers: c_i ~ Uniform(0, 1) for each RBF 
    - Widths: gamma_i ~ LogUniform(5, 50) for each RBF
    
    Input types:
        - Latent vector z: Union[np.ndarray, torch.Tensor] of shape (d * 3,)
          where d is the number of RBF basis functions
    
    Output types:
        - Full parameter vector: torch.Tensor of shape (D,) where D is the
          total number of model parameters
    """
    
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, d: int, device: str = 'cuda') -> None:
        super().__init__(model, criterion, d, device)
        
        self.d = d
        self.weight_coords = torch.linspace(0, 1, self.D).unsqueeze(1)  # shape: (N, 1)
        self.coord_dim = 1
        self.num_dims = d * (self.coord_dim + 2)

    def init_x0(self) -> np.ndarray:
        """
        Initialize RBF parameters with appropriate distributions.
        
        Returns:
            Flattened parameter vector of shape (d * 3,) containing
            [amplitude, center, gamma] for each of the d RBF kernels
        """
        # a_i ~ N(0, 0.5)
        amplitudes = np.random.randn(self.d) * 0.5

        # c_i ~ Uniform(0, 1)
        centers = np.random.rand(self.d)

        # gamma_i ~ LogUniform(5, 50)
        log_gamma = np.random.rand(self.d) * (np.log(50.0) - np.log(5.0)) + np.log(5.0)
        gammas = np.exp(log_gamma)

        # Stack into shape (K, 3)
        z = np.stack([amplitudes, centers, gammas], axis=1)
        self.x0 = z.reshape(-1)

        return z
        
    def expand(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent RBF parameters to full parameter space using Gaussian RBF kernels.
        
        Args:
            z: Latent vector of shape (d * 3,) containing flattened RBF parameters
               [a_1, c_1, gamma_1, a_2, c_2, gamma_2, ...] for d basis functions
        
        Returns:
            Full parameter tensor of shape (D,) computed as:
            theta_0 + sum_i(a_i * exp(-gamma_i * ||x - c_i||^2))
        """
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z).float()
        d, p, D = self.d, self.coord_dim, self.D
        z = z.view(d, p + 2)

        a = z[:, 0]               # (d,) - amplitudes
        c = z[:, 1:1 + p]         # (d, p) - centers
        gamma = z[:, -1]          # (d,) - widths

        # Compute squared distances: (d, D)
        dists_sq = ((c[:, None, :] - self.weight_coords[None, :, :]) ** 2).sum(dim=-1)

        # RBF evaluation: (d, D)
        phi = torch.exp(-gamma[:, None] * dists_sq)

        # Weighted sum: (D,) = sum over d basis functions
        weights = torch.sum(a[:, None] * phi, dim=0)

        return self.theta_0 + weights
    

class RandomProjectionSoftSharing(ParameterSharing):
    """
    Parameter sharing using random projection matrices.
    
    This class implements soft parameter sharing by projecting from a low-dimensional
    latent space to the full parameter space using a random projection matrix P.
    The mapping is: theta = theta_0 + P @ z, where P is a random matrix of shape (D, d).
    
    The projection matrix can optionally be normalized to have unit column norms
    and scaled by 1/sqrt(d) to maintain reasonable parameter magnitudes.
    
    Input types:
        - Latent vector z: Union[np.ndarray, torch.Tensor] of shape (d,)
    
    Output types:
        - Full parameter vector: torch.Tensor of shape (D,) where D is the
          total number of model parameters
    """
    
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, d: int, normalize: bool = False, device: str = 'cuda') -> None:
        super().__init__(model, criterion, d, device)
        self.P = None
        self.normalize = normalize
        self.init()

    def init(self) -> torch.Tensor:
        """
        Initialize the random projection matrix.
        
        Returns:
            Random projection matrix P of shape (D, d)
        """
        P = torch.randn(self.D, self.d, device=self.device)
        if self.normalize:
            P = P / P.norm(dim=0, keepdim=True)
            P = P / (self.d ** 0.5)
        self.P = P
        return P

    def expand(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using random projection.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,) computed as theta_0 + P @ z
        """
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z).float()
        z = z.to(self.device)
        return self.theta_0 + (self.P @ z)
    
    def process(self, x: Union[np.ndarray, torch.Tensor], alpha: float = 1.0) -> torch.Tensor:
        """
        Process expanded parameters with scaling.
        
        Args:
            x: Expanded parameter vector of shape (D,)
            alpha: Scaling factor for the parameters
            
        Returns:
            Processed parameter tensor of shape (D,)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
            
        x = x.to(self.device)
        
        theta = alpha * x
        
        return theta

class SparseRandomProjectionSoftSharing(ParameterSharing):
    """
    Parameter sharing using sparse random projection matrices.
    
    This class implements a sparse variant of random projection where most entries
    in the projection matrix are zero, with non-zero entries being +1 or -1.
    This follows the sparse random projection theory, where sparsity can be
    introduced without significantly affecting the projection quality.
    
    The sparsity pattern is controlled by setting entries to +sqrt(D) or -sqrt(D)
    with probability 1/sqrt(D), and 0 otherwise. The resulting matrix is then
    normalized to have unit column norms.
    
    Input types:
        - Latent vector z: Union[np.ndarray, torch.Tensor] of shape (d,)
    
    Output types:
        - Full parameter vector: torch.Tensor of shape (D,) where D is the
          total number of model parameters
    """
    
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, d: int, device: str = 'cuda') -> None:
        super().__init__(model, criterion, d, device)

        P = torch.randn(self.D, self.d, device=self.device)
        s = self.D ** 0.5 # sparsity parameter
        prob_nonzero = 1/np.sqrt(self.D) # probability of setting the column to nonzero
        mask = torch.rand(self.D, self.d) < prob_nonzero # mask of columns to set to 0
        signs = torch.randint(0, 2, (self.D, self.d), device=self.device) * 2 - 1  # +1 or -1
        P[mask] = signs[mask] * s # set the column to +1 or -1 with probability 1/sqrt(D)
        self.P = P / P.norm(dim=0, keepdim=True)

    def expand(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using sparse random projection.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,) computed as theta_0 + P @ z
            where P is a sparse random projection matrix
        """
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z).float()
        z = z.to(self.device)
        return self.theta_0 + self.P @ z
            

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
        - Latent vector z: Union[np.ndarray, torch.Tensor] of shape (d,)
    
    Output types:
        - Full parameter vector: torch.Tensor of shape (D,) where D is the
          total number of model parameters
    """
    
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, d: int, 
                 hidden_dims: List[int] = [32, 16], use_activation: bool = True, 
                 activation: str = 'relu', device: str = 'cuda') -> None:
        super().__init__(model, criterion, d, device)

        self.hidden_dims = hidden_dims
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Activation function {activation} not supported")

        layers = []
        # First layer: input to first hidden
        layers.append(nn.Linear(self.d, hidden_dims[0], bias=False))
        if use_activation:
            layers.append(self.activation)
        
        # Intermediate hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False))
            if use_activation:
                layers.append(self.activation)
        
        # Final layer: last hidden to output (no activation, as parameters can be any real values)
        layers.append(nn.Linear(hidden_dims[-1], self.D, bias=False))
        
        self.decoder = nn.Sequential(*layers)
        self.decoder.to(self.device)
        self.init_decoder()
        print("Decoder parameters number:", sum(p.numel() for p in self.decoder.parameters()))

    def init_decoder(self) -> None:
        """
        Initialize decoder weights using He normal initialization.
        
        Each layer is initialized with standard deviation sqrt(1/fan_in)
        to maintain reasonable activation magnitudes.
        """
        for layer in self.decoder.parameters():
            if isinstance(layer, nn.Linear):
                size = layer.weight.size(0)
                std = np.sqrt(1/size)
                nn.init.normal_(layer.weight, mean=0.0, std=std)

    def expand(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using MLP decoder.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,) computed as theta_0 + MLP(z)
        """
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z).float()
        z = z.to(self.device)
        with torch.no_grad():
            x = self.decoder(z)

        return self.theta_0 + x
    
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
        - Latent vector z: Union[np.ndarray, torch.Tensor] of shape (d,)
    
    Output types:
        - Full parameter vector: torch.Tensor of shape (D,) where D is the
          total number of model parameters across all layers
    """
    
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, d: int, 
                 hidden_dims: List[int] = [32, 16], emb_init: str = 'sinusoidal', 
                 device: str = 'cuda') -> None:
        super().__init__(model, criterion, d, device)

        self.d = d
        self.hidden_dims = hidden_dims
        
        layer_sizes = []
        for layer in model.parameters():
            layer_sizes.append(layer.numel())

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.embed_dim = 16

        self.embeddings = nn.Parameter(torch.empty(self.num_layers, self.embed_dim))
        if emb_init == "random":
            # Option 1: Random initialization
            nn.init.normal_(self.embeddings, mean=0.0, std=0.1)
        elif emb_init == "sinusoidal":
            # Precompute sinusoidal embeddings for all layers
            positions = torch.arange(self.num_layers, dtype=torch.float32) / (self.num_layers - 1)  # Normalized [0,1]
            freqs = torch.arange(self.embed_dim // 2, dtype=torch.float32)  # Half for sin, half for cos
            freqs = 2.0 ** freqs * torch.pi  # Exponentially increasing frequencies
            pos = positions.unsqueeze(1)  # Shape: (num_layers, 1)
            freq = freqs.unsqueeze(0)  # Shape: (1, embed_dim//2)
            sin = torch.sin(pos * freq)  # Shape: (num_layers, embed_dim//2)
            cos = torch.cos(pos * freq)  # Shape: (num_layers, embed_dim//2)
            self.embeddings = torch.cat([sin, cos], dim=1)  # Shape: (num_layers, embed_dim)
        else:
            raise ValueError("init_type must be 'random' or 'sinusoidal'")
        self.embeddings = self.embeddings.to(self.device)

        # Shared MLP: input = d + embed_dim, hidden layers, output = max(layer_sizes)
        max_output_size = max(layer_sizes)
        self.max_output_size = max_output_size
        self.decoder = nn.Sequential(
            nn.Linear(self.d + self.embed_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], max_output_size)
        )
        self.decoder.to(self.device)
        print("Decoder parameters number:", sum(p.numel() for p in self.decoder.parameters()))

    def expand(self, z: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Map latent vector to full parameter space using layer-specific hypernetwork.
        
        Args:
            z: Latent vector of shape (d,)
            
        Returns:
            Full parameter tensor of shape (D,) computed as:
            theta_0 + concatenation of layer-specific parameters generated by
            the hypernetwork conditioned on layer embeddings
        """
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z).float()
        z = z.to(self.device)
        z = z.unsqueeze(0)
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

        return self.theta_0 + torch.cat(outputs, dim=1).squeeze(0)