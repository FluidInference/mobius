"""PyTorch module for PLDA transformation, convertible to CoreML."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.linalg import eigh


class PLDATransformModule(nn.Module):
    """PLDA transformation as a CoreML-compatible PyTorch module.
    
    Applies x-vector whitening/centering followed by PLDA projection.
    """
    
    def __init__(
        self,
        mean1: np.ndarray,
        mean2: np.ndarray,
        lda: np.ndarray,
        mu: np.ndarray,
        plda_tr: np.ndarray,
        phi: np.ndarray,
        lda_dim: int = 128,
    ):
        super().__init__()
        
        # X-vector transform parameters
        self.register_buffer("mean1", torch.from_numpy(mean1.astype(np.float32)))
        self.register_buffer("mean2", torch.from_numpy(mean2.astype(np.float32)))
        self.register_buffer("lda", torch.from_numpy(lda.astype(np.float32)))
        self.register_buffer("lda_scale", torch.tensor(np.sqrt(lda.shape[0]), dtype=torch.float32))
        
        # PLDA parameters
        self.register_buffer("mu", torch.from_numpy(mu.astype(np.float32)))
        self.register_buffer("plda_tr", torch.from_numpy(plda_tr[:lda_dim, :].astype(np.float32)))
        self.register_buffer("phi", torch.from_numpy(phi[:lda_dim].astype(np.float32)))
        self.register_buffer("lda_dim_scale", torch.tensor(np.sqrt(lda_dim), dtype=torch.float32))
        
    def _l2_normalize(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
        """L2 normalize along specified dimension."""
        norm = torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim, keepdim=True), min=eps))
        return x / norm
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply x-vector transform + PLDA projection.
        
        Args:
            embeddings: Input embeddings of shape (batch, 256)
            
        Returns:
            PLDA-transformed features of shape (batch, lda_dim)
        """
        # X-vector transform
        # 1. Center and L2 normalize
        centered = embeddings - self.mean1
        normalized1 = self._l2_normalize(centered)
        
        # 2. LDA projection with scaling
        projected = torch.matmul(normalized1, self.lda) * self.lda_scale
        
        # 3. Shift and L2 normalize
        shifted = projected - self.mean2
        normalized2 = self._l2_normalize(shifted) * self.lda_dim_scale
        
        # PLDA transform
        # 1. Center with PLDA mean
        plda_centered = normalized2 - self.mu
        
        # 2. Project with PLDA transform (already truncated to lda_dim)
        plda_features = torch.matmul(plda_centered, self.plda_tr.t())
        
        return plda_features


class PLDARhoModule(nn.Module):
    """PLDA rho computation (features scaled by sqrt(phi)) for similarity scoring."""
    
    def __init__(
        self,
        mean1: np.ndarray,
        mean2: np.ndarray,
        lda: np.ndarray,
        mu: np.ndarray,
        plda_tr: np.ndarray,
        phi: np.ndarray,
        lda_dim: int = 128,
    ):
        super().__init__()
        self.transform = PLDATransformModule(mean1, mean2, lda, mu, plda_tr, phi, lda_dim)
        self.register_buffer("sqrt_phi", torch.from_numpy(np.sqrt(phi[:lda_dim]).astype(np.float32)))
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply PLDA transform and scale by sqrt(phi).
        
        Args:
            embeddings: Input embeddings of shape (batch, 256)
            
        Returns:
            Rho features of shape (batch, lda_dim) ready for dot product scoring
        """
        features = self.transform(embeddings)
        return features * self.sqrt_phi


def load_plda_module_from_npz(model_root: Path, lda_dim: int = 128) -> PLDATransformModule:
    """Load PLDA module from NPZ files."""
    transform_npz = np.load(model_root / "plda" / "xvec_transform.npz")
    plda_npz = np.load(model_root / "plda" / "plda.npz")
    
    mean1 = np.asarray(transform_npz["mean1"], dtype=np.float64)
    mean2 = np.asarray(transform_npz["mean2"], dtype=np.float64)
    lda = np.asarray(transform_npz["lda"], dtype=np.float64)
    mu = np.asarray(plda_npz["mu"], dtype=np.float64)
    tr = np.asarray(plda_npz["tr"], dtype=np.float64)
    psi = np.asarray(plda_npz["psi"], dtype=np.float64)
    
    # Compute PLDA eigendecomposition
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        w_matrix = np.linalg.inv(tr.T @ tr)
        b_matrix = np.linalg.inv((tr.T / psi).dot(tr))
        eigenvalues, eigenvectors = eigh(b_matrix, w_matrix)
    
    plda_psi = eigenvalues[::-1]
    plda_tr = eigenvectors.T[::-1]
    
    return PLDATransformModule(
        mean1=mean1,
        mean2=mean2,
        lda=lda,
        mu=mu,
        plda_tr=plda_tr,
        phi=plda_psi,
        lda_dim=lda_dim,
    )


def load_plda_rho_module_from_npz(model_root: Path, lda_dim: int = 128) -> PLDARhoModule:
    """Load PLDA rho module from NPZ files."""
    transform_npz = np.load(model_root / "plda" / "xvec_transform.npz")
    plda_npz = np.load(model_root / "plda" / "plda.npz")
    
    mean1 = np.asarray(transform_npz["mean1"], dtype=np.float64)
    mean2 = np.asarray(transform_npz["mean2"], dtype=np.float64)
    lda = np.asarray(transform_npz["lda"], dtype=np.float64)
    mu = np.asarray(plda_npz["mu"], dtype=np.float64)
    tr = np.asarray(plda_npz["tr"], dtype=np.float64)
    psi = np.asarray(plda_npz["psi"], dtype=np.float64)
    
    # Compute PLDA eigendecomposition
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        w_matrix = np.linalg.inv(tr.T @ tr)
        b_matrix = np.linalg.inv((tr.T / psi).dot(tr))
        eigenvalues, eigenvectors = eigh(b_matrix, w_matrix)
    
    plda_psi = eigenvalues[::-1]
    plda_tr = eigenvectors.T[::-1]
    
    return PLDARhoModule(
        mean1=mean1,
        mean2=mean2,
        lda=lda,
        mu=mu,
        plda_tr=plda_tr,
        phi=plda_psi,
        lda_dim=lda_dim,
    )
