#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode error at line {line_no}: {e}", file=sys.stderr)
                raise
            records.append(obj)
    return records


def convert(in_path: str, out_path: str, messages_key: str, tools_key: str, thinking_key: str) -> None:
    data = read_jsonl(in_path)
    rows: List[Dict[str, Any]] = []
    for idx, obj in enumerate(data):
        if messages_key not in obj:
            raise KeyError(f"Record {idx} missing '{messages_key}' key")
        row: Dict[str, Any] = {messages_key: obj[messages_key]}
        if tools_key in obj:
            row[tools_key] = obj[tools_key]
        if thinking_key in obj:
            row[thinking_key] = obj[thinking_key]
        rows.append(row)
    df = pd.DataFrame(rows)
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Convert JSONL multi-turn conversations to Parquet for VERL SFT")
    ap.add_argument("input_jsonl", help="Path to input JSONL file containing records with a messages array")
    ap.add_argument("output_parquet", help="Path to output Parquet file")
    ap.add_argument("--messages_key", default="messages", help="Key name for messages array (default: messages)")
    ap.add_argument("--tools_key", default="tools", help="Key name for tools field if present (default: tools)")
    ap.add_argument("--thinking_key", default="enable_thinking", help="Key name for enable_thinking if present")
    args = ap.parse_args()

    convert(args.input_jsonl, args.output_parquet, args.messages_key, args.tools_key, args.thinking_key)


if __name__ == "__main__":
    main() 