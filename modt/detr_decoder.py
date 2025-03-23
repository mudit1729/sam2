# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

class DETRDecoder(nn.Module):
    """
    DETR-style transformer decoder for object detection.
    
    This module uses object queries to extract detection features from 
    memory-conditioned features through cross-attention and self-attention.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        num_queries: int = 100,
        hidden_dim: int = 256,
        norm_first: bool = False,
    ):
        """
        Initialize the DETR decoder.
        
        Args:
            embed_dim: Dimension of the input features
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            dropout: Dropout rate
            num_queries: Number of object queries
            hidden_dim: Dimension of the decoder 
            norm_first: Whether to normalize before the attention layers
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Input projection if dimensions don't match
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Initialize object queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=norm_first,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        memory: torch.Tensor = None,
        pos_embed: torch.Tensor = None,
        query_pos: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass of the DETR decoder.
        
        Args:
            x: Memory-conditioned features - can be a tensor [B, H*W, C] 
               or a dict with 'patch_features' key
            memory: Additional memory for cross-attention (optional)
            pos_embed: Position embeddings for keys [B, H*W, C] (optional)
            query_pos: Position embeddings for queries (optional)
            
        Returns:
            Detection features [B, num_queries, hidden_dim]
        """
        # Handle input as dict (from image encoder)
        if isinstance(x, dict):
            if 'patch_features' in x:
                x = x['patch_features']
            else:
                # Debug info to help diagnose format issues
                available_keys = list(x.keys()) if hasattr(x, 'keys') else "no keys (not a dict)"
                raise ValueError(f"Input dict must contain 'patch_features' key. Available keys: {available_keys}")
        
        # Ensure x is a tensor at this point
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected x to be a tensor after extracting 'patch_features', got {type(x)}")
        
        batch_size = x.shape[0]
        
        # Project input features if needed
        memory = self.input_proj(x)
        
        # Expand object queries to batch size
        query = self.query_embed.expand(batch_size, -1, -1)
        if query_pos is None:
            query_pos = torch.zeros_like(query)
        
        # Add position embeddings
        if pos_embed is not None:
            memory = memory + pos_embed
        
        # Apply decoder layers
        # The PyTorch TransformerDecoder doesn't accept pos and query_pos directly
        # We need to add these to the inputs instead
        tgt = query
        if query_pos is not None:
            tgt = tgt + query_pos
            
        # Standard PyTorch TransformerDecoder forward call 
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
        )
        
        # Apply final projection and normalization
        output = self.norm(self.output_proj(output))
        
        return output