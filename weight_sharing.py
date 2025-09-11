import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.fft import irfft, fft
import torch
import torch.nn as nn

from utils import params_to_vector



class ParameterSharing:
    def __init__(self, model, criterion, d, device='cuda', seed=42):
        """
        Initialize parameter sharing.
        
        Args:
            model: PyTorch model (e.g., ResNet-18).
            initial_weights: Initial model weights as a flat numpy array.
            d: Number of parameters to share (K).
            device: Device for PyTorch computations.
            seed: Random seed for reproducibility.
        """
        self.device = device
        self.model = model
        self.theta_0 = params_to_vector(self.model.parameters())
        self.D = len(self.theta_0)
        self.d = d
        self.criterion = criterion

        self.assignments = np.random.randint(0, self.d, (self.D,))

        self.x0 = self.init_x0()

    def init_x0(self):
        return np.zeros(self.d)

    def set_model(self, model):
        self.model = model
        self.theta_0 = params_to_vector(self.model.parameters())
        self.D = len(self.theta_0)

    def set_theta(self, theta):
        self.theta_0 = theta

    def expand(self, z):
        """
        Assign weights to the model parameters.
        """
        x = z[self.assignments].copy()
        return x
    
    def process(self, x, alpha=1.0, max_norm=1.0):
        """
        Update theta using the mapped parameters directly.
        
        Args:
            x: D-dimensional parameter vector from lift.
            alpha: Scaling factor for the parameters.
        Returns:
            theta: D-dimensional parameter vector.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x).to(self.device).float()
            
        x = x.to(self.device)

        # x_norm = torch.norm(x)
        # if x_norm > max_norm:
        #     x = x / x_norm
        
        theta = alpha * x
        
        return theta
    
    def load_to_model(self, theta):
        """
        Set model weights from a flat numpy array.
        """
        if not isinstance(theta, torch.Tensor):
            theta = torch.from_numpy(theta).to(self.device).float()
            
        theta = theta.to(self.device)
        
        torch.nn.utils.vector_to_parameters(theta, self.model.parameters())


class GaussianRBFParameterSharing(ParameterSharing):
    def __init__(self, model, criterion, d, device='cuda', seed=42):
        super().__init__(model, criterion, d, device, seed)
        
        self.d = d
        self.weight_coords = torch.linspace(0, 1, self.D).unsqueeze(1)  # shape: (N, 1)
        self.coord_dim = 1
        self.num_dims = d * (self.coord_dim + 2)

    def init_x0(self):
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
        
    def expand(self, z):
        """
        Args:
            z (Tensor): Latent vector of shape (K * (coord_dim + 2),)
                        Contains [a_i, c_i (vector), gamma_i] for each basis function
        Returns:
            weights (Tensor): Tensor of shape (N,) with generated weights
        """
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z).float()
        d, p, D = self.d, self.coord_dim, self.D
        z = z.view(d, p + 2)

        a = z[:, 0]               # (d,)
        c = z[:, 1:1 + p]         # (d, p)
        gamma = z[:, -1]          # (d,)

        # Compute squared distances: (K, N)
        dists_sq = ((c[:, None, :] - self.weight_coords[None, :, :]) ** 2).sum(dim=-1)  # (d, D)

        # RBF evaluation
        phi = torch.exp(-gamma[:, None] * dists_sq)  # (d, D)

        # Weighted sum: (N,) = sum over K
        weights = torch.sum(a[:, None] * phi, dim=0)  # (D,)    

        return self.theta_0 + weights
    

class RandomProjectionSoftSharing(ParameterSharing):
    def __init__(self, model, criterion, d, device='cuda', seed=42):
        super().__init__(model, criterion, d, device, seed)

        P = torch.randn(self.D, self.d, device=self.device)
        self.P = P / P.norm(dim=0, keepdim=True)

    def expand(self, z):
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z).float()
        z = z.to(self.device)
        return self.theta_0 + (self.P @ z)

class SparseRandomProjectionSoftSharing(ParameterSharing):
    def __init__(self, model, criterion, d, device='cuda', seed=42):
        super().__init__(model, criterion, d, device, seed)

        P = torch.randn(self.D, self.d, device=self.device)
        s = self.D ** 0.5 # sparsity parameter
        prob_nonzero = 1/np.sqrt(self.D) # probability of setting the column to nonzero
        mask = torch.rand(self.D, self.d) < prob_nonzero # mask of columns to set to 0
        signs = torch.randint(0, 2, (self.D, self.d), device=self.device) * 2 - 1  # +1 or -1
        P[mask] = signs[mask] * s # set the column to +1 or -1 with probability 1/sqrt(D)
        self.P = P / P.norm(dim=0, keepdim=True)

    def expand(self, z):
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z).float()
        z = z.to(self.device)
        return self.theta_0 + self.P @ z
            

class MLPSoftSharing(ParameterSharing):
    def __init__(self, model, criterion, d, hidden_dims: list[int] = [32, 16], use_activation: bool = True, activation: str = 'relu', device='cuda', seed=42):
        super().__init__(model, criterion, d, device, seed)

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

    def init_decoder(self):
        for layer in self.decoder.parameters():
            if isinstance(layer, nn.Linear):
                size = layer.weight.size(0)
                std = np.sqrt(1/size)
                nn.init.normal_(layer.weight, mean=0.0, std=std)

    def expand(self, z):
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z).float()
        z = z.to(self.device)
        with torch.no_grad():
            x = self.decoder(z)

        return self.theta_0 + x
    
class HyperNetworkSoftSharing(ParameterSharing):
    def __init__(self, model, criterion, d, hidden_dims: list[int] = [32, 16], emb_init: str = 'sinusoidal', device='cuda', seed=42):
        super().__init__(model, criterion, d, device, seed)

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

        # Shared MLP: input = Z + embed_dim, hidden layers, output = max(layer_sizes)
        max_output_size = max(layer_sizes)
        self.max_output_size = max_output_size
        self.decoder = nn.Sequential(
            nn.Linear(self.K + self.embed_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], max_output_size)
        )
        self.decoder.to(self.device)
        print("Decoder parameters number:", sum(p.numel() for p in self.decoder.parameters()))

    def expand(self, z):
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z).float()
        z = z.to(self.device)
        z = z.unsqueeze(0)
        batch_size = z.size(0)
        
        # Expand z to [batch_size, num_layers, input_dim]
        z_expanded = z.unsqueeze(1).expand(-1, self.num_layers, -1)
        
        # Expand layer embeddings to [batch_size, num_layers, embed_dim]
        embeds = self.embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate: [batch_size, num_layers, input_dim + embed_dim]
        mlp_input = torch.cat([z_expanded, embeds], dim=-1)
        
        # Flatten for MLP: [batch_size * num_layers, input_dim + embed_dim]
        mlp_input = mlp_input.view(-1, mlp_input.size(-1))
        
        # Single MLP call: [batch_size * num_layers, max_output_size]
        params = self.decoder(mlp_input)
        
        # Reshape and split into list of [batch_size, s_i]
        params = params.view(batch_size, self.num_layers, self.max_output_size)
        outputs = [params[:, i, :self.layer_sizes[i]] for i in range(self.num_layers)]

        return self.theta_0 + torch.cat(outputs, dim=1).squeeze(0)