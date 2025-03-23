# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List
from torch import Tensor


class DETRDecoderLayer(nn.Module):
    """
    Transformer decoder layer for DETR-style processing.
    Performs self-attention on queries, then cross-attention with memory features,
    followed by a feedforward network.
    """
    def __init__(
        self, 
        d_model=256, 
        nhead=8,
        dim_feedforward=2048, 
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Self-attention on queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Cross-attention between queries and memory
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,  # Query embeddings [num_queries, batch_size, dim]
        memory,  # Memory from encoder [HW, batch_size, dim]
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,  # Positional encoding for memory
        query_pos: Optional[Tensor] = None,  # Positional encoding for queries
    ):
        # Self-attention among queries
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention between queries and memory
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        # Pre-normalization version
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class DETRDecoder(nn.Module):
    """
    DETR-style transformer decoder that uses object queries to extract features from encoder memory.
    """
    def __init__(
        self, 
        decoder_layer, 
        num_layers, 
        norm=None, 
        return_intermediate=False,
        d_model=256,
        num_queries=100,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        
        # Learned object queries
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.d_model = d_model

    def forward(
        self, 
        memory,  # Memory from encoder [HW, batch_size, dim]
        memory_pos=None,  # Position encoding for memory
        query_embed=None,  # Optional external queries
        memory_key_padding_mask=None,  # Padding mask for memory
    ):
        """
        Args:
            memory: encoder output features (spatial features)
            memory_pos: positional encoding for memory (spatial positions)
            query_embed: optional external object queries to use instead of learned ones
            memory_key_padding_mask: padding mask for memory
            
        Returns:
            decoder output and updated object queries
        """
        # Get batch size from memory
        batch_size = memory.shape[1]
        
        # Use provided queries or learned ones
        if query_embed is None:
            # Use learned object queries
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
            # [num_queries, batch_size, dim]
        else:
            # Ensure query_embed is shaped correctly [num_queries, batch_size, dim]
            if query_embed.dim() == 3 and query_embed.shape[1] == batch_size:
                pass  # Already in the right shape
            elif query_embed.dim() == 2:
                # Expand for batch size [num_queries, dim] -> [num_queries, batch_size, dim]
                query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
            else:
                raise ValueError(f"Unexpected query_embed shape: {query_embed.shape}")
        
        # Initialize query content with zeros
        tgt = torch.zeros_like(query_embed)
        
        # For storing intermediate outputs
        intermediate = []
        
        # Process through decoder layers
        for layer in self.layers:
            tgt = layer(
                tgt=tgt,
                memory=memory,
                memory_mask=None,
                tgt_mask=None,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=None,
                pos=memory_pos,
                query_pos=query_embed
            )
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(tgt))
                else:
                    intermediate.append(tgt)
                    
        # Apply final normalization if needed
        if self.norm is not None:
            tgt = self.norm(tgt)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(tgt)
                
        if self.return_intermediate:
            return torch.stack(intermediate)
            
        # Output shape: [num_queries, batch_size, dim]
        return tgt


def _get_clones(module, N):
    """Create N copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def build_detr_decoder(
    d_model=256, 
    nhead=8, 
    num_decoder_layers=6, 
    dim_feedforward=2048, 
    dropout=0.1,
    activation="relu",
    return_intermediate=False,
    num_queries=100,
):
    """Build a DETR decoder."""
    decoder_layer = DETRDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
    )
    
    decoder_norm = nn.LayerNorm(d_model)
    
    decoder = DETRDecoder(
        decoder_layer=decoder_layer,
        num_layers=num_decoder_layers,
        norm=decoder_norm,
        return_intermediate=return_intermediate,
        d_model=d_model,
        num_queries=num_queries,
    )
    
    return decoder