
# Qdrant Custom Sharding Benchmark

## Goal

Compare retrieval latency on a single Qdrant server.

Configuration A:
- Collection: benchmark_custom_one_shard
- Custom Sharding
- One shard key: all_teams
- All 10,000 points stored together

Configuration B:
- Collection: benchmark_custom_three_shards
- Custom Sharding
- Three shard keys:
  - hamta
  - karbordi
  - samin
- Each team stored in its own shard

No automatic sharding.

## Dataset

10,000 synthetic Persian chunks.

Metadata:
- team: hamta, karbordi, samin
- classification: normal, confidential

Vectors generated using:
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Both collections contain identical data and embeddings.

## Installation on Windows

Requirements:
- Windows 10/11
- Docker Desktop with WSL2
- Python 3.11 or 3.12
- Internet access for downloading embedding model

Start Qdrant:

docker compose up -d

Create Python environment:

py -3.11 -m venv .venv

Activate:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Run dashboard:

streamlit run app.py

Open:

http://localhost:8501

Qdrant dashboard:

http://localhost:6333/dashboard

## Initialize

Click "Initialize if missing" in Streamlit.

This operation:
1. Generates 10,000 Persian chunks.
2. Loads the embedding model.
3. Generates sentence embeddings.
4. Creates two custom-sharded collections.
5. Creates shard keys.
6. Creates payload indexes.
7. Inserts identical data into both collections.

## Benchmark

Run the following scenarios:

1. No metadata filters
2. Team filter only
3. Classification filter only
4. Team + Classification

Recommended initial configuration:

Top K: 10
Queries: 100
Warm-up: 10

Review:
- Mean latency
- P50
- P95
- P99
- Latency distribution
- Paired latency differences
- Top-K overlap

Repeat tests to reduce random variation.

## Interpretation

The three-shard configuration routes team-specific
queries directly to the corresponding team shard.

The single-shard configuration uses a shared shard
and filters the same data using Payload Index.

The benchmark measures the combination of custom
sharding and team-based query routing.

It does not independently isolate shard count
from routing and data organization.

Client timing excludes embedding and rendering.

10,000 points might not be enough to expose
meaningful differences in HNSW performance.

## Reset

Use "Rebuild benchmark collections" in Streamlit.

This deletes and recreates only the two
benchmark collections.

To stop Qdrant:

docker compose down

Do not use "docker compose down -v" unless
you intentionally want to delete its storage volume.
