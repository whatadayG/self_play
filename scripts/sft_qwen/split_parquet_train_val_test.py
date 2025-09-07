#!/usr/bin/env python3
import argparse
import os
import math
import pandas as pd


def split_parquet(input_parquet: str, output_dir: str, train_frac: float, val_frac: float, test_frac: float, seed: int = 42):
    total = train_frac + val_frac + test_frac
    if not math.isclose(total, 1.0, rel_tol=1e-6):
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_parquet(input_parquet)
    n = len(df)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # Ensure total counts sum to n
    n_test = n - n_train - n_val

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    base = os.path.splitext(os.path.basename(input_parquet))[0]

    train_path = os.path.join(output_dir, f"{base}_train.parquet")
    val_path = os.path.join(output_dir, f"{base}_val.parquet")
    test_path = os.path.join(output_dir, f"{base}_test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Input: {input_parquet} (rows={n})")
    print(f"Wrote train: {train_path} (rows={len(train_df)})")
    print(f"Wrote  val : {val_path} (rows={len(val_df)})")
    print(f"Wrote test: {test_path} (rows={len(test_df)})")

    return train_path, val_path, test_path


def main():
    ap = argparse.ArgumentParser(description="Split a Parquet dataset into train/val/test Parquet files")
    ap.add_argument("input_parquet", help="Path to input Parquet file")
    ap.add_argument("output_dir", help="Directory to write split Parquet files")
    ap.add_argument("--train_frac", type=float, default=0.9, help="Train fraction (default 0.9)")
    ap.add_argument("--val_frac", type=float, default=0.05, help="Validation fraction (default 0.05)")
    ap.add_argument("--test_frac", type=float, default=0.05, help="Test fraction (default 0.05)")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed (default 42)")
    args = ap.parse_args()

    split_parquet(args.input_parquet, args.output_dir, args.train_frac, args.val_frac, args.test_frac, args.seed)


if __name__ == "__main__":
    main() 