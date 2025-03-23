# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedMemoryAttention(nn.Module):
    """
    Memory attention module that conditions current frame features with memory vectors.
    
    This module implements a cross-attention mechanism between current frame features
    and memory vectors from previous frames to enhance temporal consistency.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        memory_dim: int = None,
        max_memory_length: int = 10,
    ):
        """
        Initialize the memory attention module.
        
        Args:
            embed_dim: Dimension of the input features
            num_heads: Number of attention heads
            dropout: Dropout rate
            memory_dim: Dimension of memory vectors (defaults to embed_dim)
            max_memory_length: Maximum number of memory frames to store
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_dim = memory_dim if memory_dim is not None else embed_dim
        self.num_heads = num_heads
        self.max_memory_length = max_memory_length
        
        # Normalize input and memory before attention
        self.norm_input = nn.LayerNorm(embed_dim)
        self.norm_memory = nn.LayerNorm(self.memory_dim)
        
        # Cross-attention between current features and memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            kdim=self.memory_dim,
            vdim=self.memory_dim,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_output = nn.LayerNorm(embed_dim)
        
        # Initialize memory bank
        self.register_buffer("empty_memory", torch.zeros(1, 1, self.memory_dim))
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the memory attention module.
        
        Args:
            x: Input features [B, N, C]
            memory: Memory vectors [B, M, C_m] (optional)
            memory_key_padding_mask: Mask for padding in memory [B, M] (optional)
            
        Returns:
            Enhanced features with memory conditioning [B, N, C]
        """
        # Normalize inputs
        x_norm = self.norm_input(x)
        
        # If no memory is provided, use empty memory
        if memory is None or memory.shape[1] == 0:
            # Return input as is if no memory is available
            return x
        else:
            memory_norm = self.norm_memory(memory)
        
        # Apply cross-attention
        attn_output, _ = self.cross_attn(
            query=x_norm,
            key=memory_norm,
            value=memory_norm,
            key_padding_mask=memory_key_padding_mask,
        )
        
        # Apply output projection, dropout, and residual connection
        output = x + self.dropout(self.output_proj(attn_output))
        output = self.norm_output(output)
        
        return output