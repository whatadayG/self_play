#!/usr/bin/env python3
# Copyright 2025

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


class PreTokenizedSFTDataset(Dataset):
    """
    Dataset for pre-tokenized SFT samples.

    Expects Parquet file(s) with the following columns (array-like of ints):
      - input_ids
      - attention_mask
      - position_ids
      - loss_mask
    Optional:
      - sample_weight (float scalar per-sample)

    All sequences are assumed to have the same padded length within a file.
    """

    def __init__(self, parquet_files, tokenizer, config):  # tokenizer kept for API parity
        if not isinstance(parquet_files, (list, tuple)):
            parquet_files = [parquet_files]

        dataframes = [pd.read_parquet(pf) for pf in parquet_files]
        self.df = pd.concat(dataframes, ignore_index=True)

        required_cols = ["input_ids", "attention_mask", "position_ids", "loss_mask"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column '{col}' in pretokenized parquet")

        self.has_weight = "sample_weight" in self.df.columns
        self.max_length = config.get("max_length", None)

    def __len__(self) -> int:
        return len(self.df)

    def _to_tensor_1d(self, x) -> torch.Tensor:
        t = torch.tensor(x, dtype=torch.long)
        return t

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        input_ids = self._to_tensor_1d(row["input_ids"])  # (L,)
        attention_mask = self._to_tensor_1d(row["attention_mask"])  # (L,)
        position_ids = self._to_tensor_1d(row["position_ids"])  # (L,) or (3,L) for some models
        loss_mask = self._to_tensor_1d(row["loss_mask"])  # (L,)

        # Optional length clamp/pad to max_length if provided
        if self.max_length is not None:
            L = input_ids.size(0)
            if L > self.max_length:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                position_ids = position_ids[: self.max_length] if position_ids.dim() == 1 else position_ids[:, : self.max_length]
                loss_mask = loss_mask[: self.max_length]
            elif L < self.max_length:
                pad_len = self.max_length - L
                pad_token_id = 0
                input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
                if position_ids.dim() == 1:
                    position_ids = torch.cat([position_ids, torch.arange(L, self.max_length, dtype=position_ids.dtype)])
                else:
                    # basic right-pad for multi-dim position_ids
                    pad_pos = torch.zeros((position_ids.size(0), pad_len), dtype=position_ids.dtype)
                    position_ids = torch.cat([position_ids, pad_pos], dim=1)
                loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=loss_mask.dtype)])

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

        if self.has_weight:
            item["sample_weight"] = float(row["sample_weight"])  # scalar

        return item


