"""Build normalized Summary embeddings from issues.csv using BGE-M3.

Outputs:
    issue_embeddings.npy
        Float32 matrix with one normalized dense embedding per valid row.
    issue_embedding_metadata.csv
        Source row, Jira Key (when present), and Summary in embedding-row order.

Example:
    python build_embeddings.py --input issues.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed the Summary column of issues.csv using BAAI/bge-m3."
    )
    parser.add_argument("--input", default="20260730_issues.csv", help="Input CSV path.")
    parser.add_argument(
        "--embeddings-output",
        default="issue_embeddings.npy",
        help="Output NumPy embedding matrix.",
    )
    parser.add_argument(
        "--metadata-output",
        default="issue_embedding_metadata.csv",
        help="Output CSV whose rows align with the embedding matrix.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Encoding batch size. Reduce this if memory is limited.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path.resolve()}")

    dataframe = pd.read_csv(input_path)
    if "Summary" not in dataframe.columns:
        raise ValueError(
            "Required column 'Summary' was not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    # Keep original CSV row numbers so the search app can align vectors safely.
    metadata = dataframe.copy()
    metadata.insert(0, "Source row", metadata.index)
    metadata["Summary"] = metadata["Summary"].fillna("").astype(str).str.strip()
    metadata = metadata.loc[metadata["Summary"].ne("")].copy()

    if metadata.empty:
        raise ValueError("No non-empty values were found in the Summary column.")

    summaries = metadata["Summary"].tolist()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")

    model = SentenceTransformer(MODEL_NAME, device=device)

    embeddings = model.encode(
        summaries,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)

    embeddings_path = Path(args.embeddings_output)
    metadata_path = Path(args.metadata_output)

    np.save(embeddings_path, embeddings)

    output_columns = ["Source row"]
    if "Key" in metadata.columns:
        output_columns.append("Key")
    output_columns.append("Summary")

    metadata[output_columns].to_csv(
        metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Saved embeddings: {embeddings_path.resolve()}")
    print(f"Saved metadata:   {metadata_path.resolve()}")
    print(f"Embedding shape:  {embeddings.shape}")


if __name__ == "__main__":
    main()