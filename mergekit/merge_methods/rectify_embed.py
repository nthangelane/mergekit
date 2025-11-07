# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: BUSL-1.1


import logging
from typing import List

import torch

from mergekit.architecture import WeightInfo


def rectify_embed_sizes(weight_info: WeightInfo, tensors: List[torch.Tensor]):
    # TODO: use arch_info.embed_weights() instead
    if not weight_info.is_embed or not all(len(t.shape) == 2 for t in tensors):
        return

    max_rows = max(t.shape[0] for t in tensors)
    max_cols = max(t.shape[1] for t in tensors)

    resized = False
    for idx, tensor in enumerate(tensors):
        rows, cols = tensor.shape
        if rows == max_rows and cols == max_cols:
            continue

        new_tensor = tensor.new_zeros((max_rows, max_cols))
        new_tensor[:rows, :cols] = tensor
        tensors[idx] = new_tensor
        resized = True

    if resized:
        logging.warning(
            f"Padded embeddings to common size {(max_rows, max_cols)} for {weight_info.name}"
        )
