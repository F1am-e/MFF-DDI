This repository contains three custom PyTorch modules designed to enhance feature representation and model robustness in deep learning architectures. These components are particularly useful for sequence processing and Graph Neural Networks (GNNs).

Components Overview

1. Spatial Group Enhancement for 1D (`GroupEnhancementModule.py`)
Class: `SpatialGroupEnhance_for_1D`

This module applies a spatial group enhancement mechanism to 1D sequence data (e.g., outputs from LSTM or CNN layers). 
Mechanism: It divides the input feature channels into multiple groups, performs global average pooling within each group, normalizes the statistics, and applies a learnable affine transformation (weight and bias). 
Output:It generates a sigmoid-activated gating mask that recalibrates the original features, enhancing semantically important group representations.

2. Link Attention (`LinkAttention.py`)
Class: `LinkAttention`

A masked attention module designed to weigh the importance of different nodes or sequence steps.
Mechanism:It projects the input to compute attention queries and applies a strict masking strategy (setting invalid/padded positions to `-9e15`). 
Output: It computes softmax-normalized attention weights to aggregate the values, returning both the weighted average representation (squeezed) and the attention maps.

3. SkipNode Mechanism (`SkipNode.py`)
Functions: `skip_node_mask`, `skip_node`

A regularization mechanism primarily used in deep Graph Neural Networks to prevent over-smoothing.
Mechanism: It randomly selects a subset of nodes based on a specified `skip_rate` and `skip_type` (either a `uniform` distribution or a `degree`-based probability). 
Output: For the selected nodes, it skips the feature update for the current layer, carrying forward their original features (`x_old`). This helps preserve local, highly-differentiated node information in deep architectures.

Dependencies
- `torch` (PyTorch)
- `torch.utils.data` (for SkipNode dataloaders)
