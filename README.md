# Jira Issue Matcher

A Python-based Jira analytics pipeline that retrieves Jira issues, filters them by **VIN** and **FRS**, generates semantic embeddings using **BGE-M3**, and clusters similar issues to identify duplicate or related vehicle defects.

This project was developed to assist vehicle testing teams in quickly finding recurring issues and reducing manual effort when reviewing Jira tickets.

---

## Features

- Download Jira issues directly through the Jira REST API
- Filter issues using predefined VIN + FRS combinations
- Export matching and non-matching issues to CSV
- Generate semantic embeddings from issue summaries using **BAAI/bge-m3**
- Cluster semantically similar issues using cosine similarity
- Restrict clustering within the same vehicle Program
- Produce human-readable cluster reports

---

## Project Workflow

```text
Recent VIN + FRS List
          │
          ▼
jira_recent_frs.py
          │
          ▼
20260730_issues.csv
          │
          ▼
build_embeddings.py
          │
          ├── issue_embeddings.npy
          └── issue_embedding_metadata.csv
                      │
                      ▼
cluster_issues.py
                      │
                      ▼
issue_clusters_by_program.txt
```

---

## Project Structure

```text
.
├── jira_recent_frs.py              # Download and filter Jira issues
├── build_embeddings.py             # Generate BGE-M3 embeddings
├── cluster_issues.py               # Cluster similar issues
│
├── 20260730_recent_frs_by_vin.txt  # Input VIN/FRS list
├── 20260730_issues.csv             # Filtered Jira issues
├── issue_embeddings.npy            # Dense embedding vectors
├── issue_embedding_metadata.csv    # Metadata aligned with embeddings
├── issue_clusters_by_program.txt   # Clustering result
│
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your_username>/jira-issue-matcher.git

cd jira-issue-matcher
```

Install dependencies

```bash
pip install -r requirements.txt
```

Example requirements:

```text
numpy
pandas
requests
torch
sentence-transformers
scikit-learn
```

---

## Step 1 – Download Jira Issues

Prepare a text file containing the VINs and FRS values you want to keep.

Example:

```text
VIN: RLNV5JSE2RH770573
1. FRS.01.02
2. FRS.01.03

VIN: RLNV5JSE2RH770574
1. FRS.03.05
```

Run

```bash
python jira_recent_frs.py
```

The script will

- Query Jira via REST API
- Match issues by VIN and FRS
- Export

```
20260730_issues.csv
```

---

## Step 2 – Generate Embeddings

Generate semantic embeddings from issue summaries.

```bash
python build_embeddings.py
```

Outputs

```
issue_embeddings.npy
issue_embedding_metadata.csv
```

Model used

```
BAAI/bge-m3
```

---

## Step 3 – Cluster Similar Issues

Run

```bash
python cluster_issues.py
```

Default similarity threshold

```
0.85
```

Example

```bash
python cluster_issues.py --threshold 0.90
```

Output

```
issue_clusters_by_program.txt
```

Example

```text
Program: VF5

Cluster 1:
VQI-12543
VQI-12851
VQI-13340

Cluster 2:
VQI-14002
VQI-14190
```

---

## Clustering Method

The project uses

- **Sentence Embeddings:** BAAI/bge-m3
- **Distance Metric:** Cosine similarity
- **Clustering Algorithm:** Agglomerative Clustering
- **Linkage:** Complete linkage

Clusters are only created when

- Every issue belongs to the same Program
- Every pair of issues satisfies the minimum cosine similarity threshold

---

## Output Files

| File | Description |
|-------|-------------|
| 20260730_issues.csv | Filtered Jira issues |
| issue_embeddings.npy | Dense embedding vectors |
| issue_embedding_metadata.csv | Mapping between embeddings and Jira issues |
| issue_clusters_by_program.txt | Final clustering result |

---

## Example Use Cases

- Detect duplicate Jira tickets
- Identify recurring vehicle defects
- Group similar customer complaints
- Analyze issue trends across Programs
- Support root cause analysis
- Reduce manual issue triage

---

## Technologies

- Python
- Jira REST API
- Pandas
- NumPy
- Sentence Transformers
- BAAI/bge-m3
- Scikit-learn
- PyTorch

---

## Notes

This repository is intended for research and internal analytics. Authentication credentials (cookies, API tokens, etc.) should **never** be committed to GitHub. Store secrets using environment variables or a `.env` file and add them to `.gitignore`.

---

## License

MIT License
