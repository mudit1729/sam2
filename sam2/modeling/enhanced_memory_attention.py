# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List, Tuple

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.sam2_utils import get_activation_fn, get_clones


class EnhancedMemoryAttentionLayer(nn.Module):
    """
    Enhanced memory attention layer that accepts previous memory vectors
    and conditions current frame features with them.
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        
        # Self attention on current features
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Cross attention between current features and memory vectors
        self.cross_attn_memory = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization and dropout layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = get_activation_fn(activation)
        self.normalize_before = normalize_before
        
    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """Add positional embeddings to tensor if provided."""
        return tensor if pos is None else tensor + pos
        
    def forward_post(
        self,
        tgt: Tensor,  # Current frame features
        memory: Tensor,  # Memory vectors from previous frames
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,  # Positional encoding for memory
        query_pos: Optional[Tensor] = None,  # Positional encoding for current features
    ) -> Tensor:
        """Post-normalization forward pass."""
        # Self attention on current features
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # If memory is provided, apply cross attention
        if memory is not None and memory.size(0) > 0:
            # Cross attention between current features and memory
            tgt2 = self.cross_attn_memory(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        
        # Feed forward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
        
    def forward_pre(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Pre-normalization forward pass."""
        # Self attention
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        
        # Cross attention
        if memory is not None and memory.size(0) > 0:
            tgt2 = self.norm2(tgt)
            tgt2 = self.cross_attn_memory(
                query=self.with_pos_embed(tgt2, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )[0]
            tgt = tgt + self.dropout2(tgt2)
        
        # Feed forward
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
        
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass based on normalization strategy."""
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask,
                pos, query_pos
            )
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask,
            pos, query_pos
        )


class EnhancedMemoryAttention(nn.Module):
    """
    Main memory attention module that applies multiple attention layers
    to condition current frame features with memory vectors.
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        batch_first: bool = True,  # Whether input is batch-first
    ):
        super().__init__()
        
        # Create attention layer
        layer = EnhancedMemoryAttentionLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
        )
        
        # Create multiple layers
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        
        self.norm = nn.LayerNorm(d_model)
        self.batch_first = batch_first
        
    def forward(
        self,
        curr_features: Tensor,  # Current frame features
        memory_vectors: Optional[Tensor] = None,  # Memory vectors from previous frames
        curr_pos: Optional[Tensor] = None,  # Positional encoding for current features
        memory_pos: Optional[Tensor] = None,  # Positional encoding for memory vectors
    ) -> Tensor:
        """
        Forward pass for enhanced memory attention.
        
        Args:
            curr_features: Current frame features [batch_size, seq_len, dim] if batch_first=True
                           else [seq_len, batch_size, dim]
            memory_vectors: Memory vectors from previous frames [num_vectors, batch_size, dim]
                           or None if this is the first frame
            curr_pos: Positional encoding for current features (same shape as curr_features)
            memory_pos: Positional encoding for memory vectors (same shape as memory_vectors)
            
        Returns:
            Memory-conditioned features (same shape as curr_features)
        """
        # Convert to sequence-first if needed
        if self.batch_first:
            curr_features = curr_features.transpose(0, 1)
            if curr_pos is not None:
                curr_pos = curr_pos.transpose(0, 1)
            
        # Process through layers
        output = curr_features
        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=memory_vectors,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None,
                pos=memory_pos,
                query_pos=curr_pos,
            )
            
        # Apply normalization
        output = self.norm(output)
        
        # Convert back to batch-first if needed
        if self.batch_first:
            output = output.transpose(0, 1)
            
        return output


def build_enhanced_memory_attention(
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 2,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: str = "relu",
    normalize_before: bool = False,
) -> EnhancedMemoryAttention:
    """Helper function to build an enhanced memory attention module."""
    return EnhancedMemoryAttention(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        normalize_before=normalize_before,
    )