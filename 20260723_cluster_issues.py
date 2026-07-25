from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster BGE-M3 issue embeddings by cosine similarity, while "
            "requiring every cluster member to have the same Program."
        )
    )
    parser.add_argument("--input", default="20260730_issues.csv", help="Input CSV path.")
    parser.add_argument(
        "--embeddings",
        default="issue_embeddings.npy",
        help="Embedding matrix created by build_embeddings.py.",
    )
    parser.add_argument(
        "--metadata",
        default="issue_embedding_metadata.csv",
        help="Embedding metadata created by build_embeddings.py.",
    )
    parser.add_argument(
        "--output",
        default="issue_clusters_by_program.txt",
        help="Output text file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum cosine similarity within a cluster (default: 0.85).",
    )
    return parser.parse_args()


def load_aligned_data(
    issues_path: Path,
    embeddings_path: Path,
    metadata_path: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    for path in (issues_path, embeddings_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path.resolve()}")

    issues = pd.read_csv(issues_path)
    metadata = pd.read_csv(metadata_path)
    embeddings = np.load(embeddings_path)

    required_issue_columns = {"Key", "Summary", "Program"}
    missing = required_issue_columns.difference(issues.columns)
    if missing:
        raise ValueError(
            f"20260730_issues.csv is missing required columns: {sorted(missing)}"
        )

    if "Source row" not in metadata.columns:
        raise ValueError(
            "Embedding metadata has no 'Source row' column. "
            "Rebuild it with build_embeddings.py."
        )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be a 2D matrix; received shape {embeddings.shape}."
        )

    if len(metadata) != len(embeddings):
        raise ValueError(
            "Embedding metadata and embedding matrix have different row counts. "
            "Rebuild embeddings before clustering."
        )

    source_rows = pd.to_numeric(
        metadata["Source row"], errors="raise"
    ).astype(int)
    if source_rows.duplicated().any():
        raise ValueError("Embedding metadata contains duplicate Source row values.")
    if (source_rows < 0).any() or (source_rows >= len(issues)).any():
        raise ValueError("Embedding metadata contains invalid Source row values.")

    aligned = issues.iloc[source_rows.to_numpy()].reset_index(drop=True).copy()

    # Catch a stale embedding/CSV combination even when row counts happen to match.
    if "Key" in metadata.columns:
        metadata_keys = metadata["Key"].fillna("").astype(str).reset_index(drop=True)
        aligned_keys = aligned["Key"].fillna("").astype(str).reset_index(drop=True)
        if not metadata_keys.equals(aligned_keys):
            raise ValueError(
                "20260730_issues.csv no longer aligns with the embedding metadata. "
                "Run build_embeddings.py again."
            )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("The embedding matrix contains a zero-length vector.")
    embeddings = embeddings / norms

    aligned["Key"] = aligned["Key"].fillna("").astype(str).str.strip()
    aligned["Program"] = (
        aligned["Program"]
        .fillna("<Missing Program>")
        .astype(str)
        .str.strip()
        .replace("", "<Missing Program>")
    )

    if aligned["Key"].eq("").any():
        raise ValueError("One or more embedded issues have an empty Key.")

    return aligned, embeddings


def cluster_one_program(
    program_embeddings: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if len(program_embeddings) == 1:
        return np.array([0], dtype=int)

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="complete",
        distance_threshold=1.0 - threshold,
    )
    return model.fit_predict(program_embeddings)


def find_clusters(
    issues: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
) -> dict[str, list[list[str]]]:
    results: dict[str, list[list[str]]] = {}

    for program, rows in issues.groupby("Program", sort=True).groups.items():
        indices = np.asarray(list(rows), dtype=int)
        labels = cluster_one_program(embeddings[indices], threshold)
        program_clusters: list[list[str]] = []

        for label in np.unique(labels):
            member_positions = np.flatnonzero(labels == label)
            if len(member_positions) < 2:
                continue

            member_indices = indices[member_positions]
            keys = issues.iloc[member_indices]["Key"].tolist()

            # Verify the promised complete-linkage condition despite floating
            # point rounding inside the clustering implementation.
            vectors = embeddings[member_indices]
            similarities = vectors @ vectors.T
            pair_scores = similarities[
                np.triu_indices(len(member_indices), k=1)
            ]
            if pair_scores.min() + 1e-6 < threshold:
                raise RuntimeError(
                    "A generated cluster failed the minimum-similarity check."
                )

            program_clusters.append(keys)

        if program_clusters:
            program_clusters.sort(key=lambda cluster: (-len(cluster), cluster))
            results[str(program)] = program_clusters

    return results


def write_text_output(
    clusters_by_program: dict[str, list[list[str]]],
    output_path: Path,
    threshold: float,
) -> tuple[int, int]:
    lines = [
        "BGE-M3 issue clusters by Program",
        f"Cosine similarity threshold: {threshold:.4f}",
        "",
    ]
    cluster_count = 0
    issue_count = 0

    if not clusters_by_program:
        lines.append("No multi-issue clusters found.")
    else:
        for program, clusters in clusters_by_program.items():
            lines.append(f"Program: {program}")
            for number, keys in enumerate(clusters, start=1):
                lines.append(f"  Cluster {number}: {', '.join(keys)}")
                cluster_count += 1
                issue_count += len(keys)
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return cluster_count, issue_count


def main() -> None:
    args = parse_args()
    if not -1.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between -1.0 and 1.0.")

    issues, embeddings = load_aligned_data(
        Path(args.input),
        Path(args.embeddings),
        Path(args.metadata),
    )
    clusters = find_clusters(issues, embeddings, args.threshold)
    cluster_count, issue_count = write_text_output(
        clusters,
        Path(args.output),
        args.threshold,
    )

    print(
        f"Found {cluster_count} clusters containing {issue_count} issues "
        f"at threshold {args.threshold:.4f}."
    )
    print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()