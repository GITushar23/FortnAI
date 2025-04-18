import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SymExpTwoHotDist:
    def __init__(self, num_bins=255, min_range=-5.0, max_range=5.0):
        self.num_bins = num_bins
        # Create bins within a reasonable range to prevent overflow
        if num_bins % 2 == 0:
            half = np.linspace(min_range, 0, num_bins // 2, dtype=np.float32)
            half = self.symexp(half)
            self.bins = np.concatenate([half, -half[::-1]])
        else:
            half = np.linspace(min_range, 0, (num_bins - 1) // 2 + 1, dtype=np.float32)
            half = self.symexp(half)
            self.bins = np.concatenate([half, -half[:-1][::-1]])
        
        self.bins = torch.FloatTensor(self.bins)
    
    @staticmethod
    def symexp(x):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1) if isinstance(x, torch.Tensor) else np.sign(x) * (np.exp(np.abs(x)) - 1)

    def preprocess_target(self, target):
        """
        Convert target values to two-hot encoding
        Args:
            target: tensor of shape [batch, time_seq_len] or [...], where ... represents any number of batch dimensions
        Returns:
            target_two_hot: tensor of shape [batch, time_seq_len, num_bins] or [..., num_bins]
        """
        # Ensure target is at least 1D
        if target.dim() == 0:
            target = target.unsqueeze(0)
            
        # Expand target to have a feature dimension if it doesn't
        if target.dim() == 1:
            target = target.unsqueeze(-1)
            
        # Get the bins ready on the correct device
        bins = self.bins.to(target.device)
        
        # Reshape target and bins for broadcasting
        # Append a new dimension for the bins
        target_shape = target.shape + (1,)  # e.g., [batch, time_seq_len, 1]
        # Prepend singleton dimensions to bins_shape to match target's dimensions
        bins_shape = (1,) * target.dim() + (self.num_bins,)  # e.g., [1, 1, num_bins]
        
        # Reshape tensors
        target = target.view(*target_shape)  # [..., 1]
        bins = bins.view(*bins_shape)        # [1, ..., num_bins]
        
        # Calculate distances
        dists = torch.abs(target - bins)  # [..., num_bins]
        
        # Find two closest bins along the last dimension
        closest_bin_idx = torch.topk(dists, k=2, dim=-1, largest=False)[1]  # [..., 2]
        
        # Gather closest distances
        closest_dists = torch.gather(dists, -1, closest_bin_idx)  # [..., 2]
        
        # Calculate weights based on distances
        weights = 1 - closest_dists / closest_dists.sum(dim=-1, keepdim=True)  # [..., 2]
        
        # Create two-hot target
        target_two_hot = torch.zeros(*target.shape[:-1], self.num_bins, device=target.device)  # [..., num_bins]
        target_two_hot.scatter_(-1, closest_bin_idx, weights)
        
        return target_two_hot


    def sample(self, logits):
        """
        Sample from the distribution
        Args:
            logits: tensor of shape [..., num_bins] where ... represents any number of batch dimensions
        Returns:
            expected_value: tensor of shape [...]
        """
        probs = F.softmax(logits, dim=-1)
        bins_expanded = self.bins.to(logits.device)
        
        # Expand bins to match logits' batch dimensions if necessary
        if logits.dim() > 1:
            bins_expanded = bins_expanded.view((1,) * (logits.dim() - 1) + (-1,))
            bins_expanded = bins_expanded.expand(logits.shape)
        
        expected_value = (probs * bins_expanded).sum(dim=-1)
        return expected_value
    
    def mode(self, logits):
        """
        Get the mode of the distribution
        Args:
            logits: tensor of shape [..., num_bins] where ... represents any number of batch dimensions
        Returns:
            mode: tensor of shape [...]
        """
        return self.bins[logits.argmax(dim=-1)]
    

    def __repr__(self):
        return f"SymExpTwoHotDist(num_bins={self.num_bins})"