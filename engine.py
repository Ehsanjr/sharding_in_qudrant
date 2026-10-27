
"""Qdrant custom-sharding latency benchmark engine."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import requests
from qdrant_client import QdrantClient, models
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer

from dataset import make_dataset, TEAMS, LEVELS


URL = os.getenv("QDRANT_URL", "http://localhost:6333")

MODEL_NAME = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

ROOT = Path(__file__).parent
CACHE = ROOT / "cache"

SINGLE = "benchmark_custom_one_shard"
THREE = "benchmark_custom_three_shards"

SHARD_SHARED = "all_teams"

N = 10000
SEED = 2026


def client():
    return QdrantClient(
        url=URL,
        timeout=120,
        check_compatibility=False,
    )


@lru_cache(maxsize=1)
def encoder():
    return SentenceTransformer(
        MODEL_NAME,
        local_files_only=True
    )


def prepare_embeddings(progress=None):
    CACHE.mkdir(exist_ok=True)

    frame = make_dataset(N, SEED)

    vfile = CACHE / "vectors.npy"
    meta = CACHE / "meta.json"

    expected = {
        "model": MODEL_NAME,
        "rows": N,
        "seed": SEED,
    }

    if vfile.exists() and meta.exists():
        saved_meta = json.loads(
            meta.read_text(encoding="utf-8")
        )

        if saved_meta == expected:
            vectors = np.load(vfile)

            if vectors.shape[0] == N:
                return frame, vectors

    if progress:
        progress(
            "Loading embedding model and encoding "
            "10,000 Persian passages..."
        )

    model = encoder()

    vectors = model.encode(
        frame["text"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    np.save(vfile, vectors)

    meta.write_text(
        json.dumps(expected),
        encoding="utf-8",
    )

    return frame, vectors


def upsert_chunked(
    qc,
    collection,
    keys,
    df,
    vectors,
    progress=None,
):
    for group, selector in keys:

        if group:
            selection = df.index[
                df["team"] == group
            ].to_numpy()
        else:
            selection = np.arange(len(df))

        for offset in range(0, len(selection), 128):
            ix = selection[offset:offset + 128]

            points = []

            for j in ix:
                row = df.iloc[j]

                points.append(
                    models.PointStruct(
                        id=int(row["id"]),
                        vector=vectors[j].tolist(),
                        payload={
                            "team": str(row["team"]),
                            "classification": str(
                                row["classification"]
                            ),
                            "text": str(row["text"]),
                        },
                    )
                )

            qc.upsert(
                collection_name=collection,
                points=points,
                shard_key_selector=selector,
                wait=True,
            )

            if progress and offset % 1280 == 0:
                progress(
                    f"{collection}: "
                    f"{min(offset + len(ix), len(selection))}"
                    f"/{len(selection)} "
                    f"in {selector}"
                )


def initialize(progress=None, recreate=False):
    qc = client()

    frame, vectors = prepare_embeddings(progress)
    dim = int(vectors.shape[1])

    configurations = [
        (SINGLE, [SHARD_SHARED]),
        (THREE, list(TEAMS)),
    ]

    for name, shard_keys in configurations:

        exists = qc.collection_exists(name)

        if exists and recreate:
            qc.delete_collection(name)
            exists = False

        if exists:
            info = qc.get_collection(name)

            if int(info.points_count or 0) == N:
                if progress:
                    progress(
                        f"{name}: already populated, skipping"
                    )
                continue

            raise RuntimeError(
                f"Collection {name} exists with "
                f"{info.points_count} points. "
                "Use 'Rebuild benchmark collections'."
            )

        if progress:
            progress(
                f"Creating {name} with "
                f"{len(shard_keys)} custom shard keys"
            )

        qc.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
            sharding_method=models.ShardingMethod.CUSTOM,
            shard_number=1,
            replication_factor=1,
            on_disk_payload=False,
        )

        for key in shard_keys:
            qc.create_shard_key(
                collection_name=name,
                shard_key=key,
            )

        for field in ("team", "classification"):
            qc.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

        if name == SINGLE:
            groups = [(None, SHARD_SHARED)]
        else:
            groups = [(t, t) for t in TEAMS]

        upsert_chunked(
            qc,
            name,
            groups,
            frame,
            vectors,
            progress,
        )

        if progress:
            progress(f"Completed {name}")

    return {
        "rows": len(frame),
        "dimension": dim,
        "model": MODEL_NAME,
    }


def get_overview():
    qc = client()
    results = []

    configs = [
        (SINGLE, "1 custom shard"),
        (THREE, "3 team-based custom shards"),
    ]

    for name, label in configs:

        if not qc.collection_exists(name):
            results.append({
                "collection": label,
                "exists": False,
            })
            continue

        info = qc.get_collection(name)

        results.append({
            "collection": label,
            "exists": True,
            "points": info.points_count,
            "segments": info.segments_count,
            "status": str(info.status),
            "optimizer_status": str(
                info.optimizer_status
            ),
            "shard_keys": (
                SHARD_SHARED
                if name == SINGLE
                else ", ".join(TEAMS)
            ),
        })

    return pd.DataFrame(results)


def build_filter(team=None, classification=None):
    conditions = []

    if team:
        conditions.append(
            models.FieldCondition(
                key="team",
                match=models.MatchValue(value=team),
            )
        )

    if classification:
        conditions.append(
            models.FieldCondition(
                key="classification",
                match=models.MatchValue(
                    value=classification
                ),
            )
        )

    return (
        models.Filter(must=conditions)
        if conditions
        else None
    )



USAGE_FIELDS = [
    "cpu",
    "vector_io_read",
    "vector_io_write",
    "payload_io_read",
    "payload_io_write",
    "payload_index_io_read",
    "payload_index_io_write",
]


def run_query(
    qc,
    collection,
    vector,
    team=None,
    classification=None,
    top_k=10,
    exact=False,
    with_payload=False,
):
    shard = (
        SHARD_SHARED
        if collection == SINGLE
        else team
    )

    body = {
        "query": (
            vector.tolist()
            if hasattr(vector, "tolist")
            else list(vector)
        ),
        "limit": int(top_k),
        "params": {"exact": bool(exact)},
        "with_payload": bool(with_payload),
        "with_vector": False,
    }

    query_filter = build_filter(
        team, classification
    )

    if query_filter is not None:
        body["filter"] = query_filter.model_dump(
            mode="json",
            exclude_none=True,
        )

    if shard is not None:
        body["shard_key"] = shard

    start = time.perf_counter_ns()

    response = requests.post(
        f"{URL.rstrip('/')}/collections/"
        f"{collection}/points/query",
        json=body,
        timeout=120,
    )

    elapsed_ms = (
        time.perf_counter_ns() - start
    ) / 1e6

    response.raise_for_status()
    data = response.json()

    if data.get("status") != "ok":
        raise RuntimeError(str(data))

    hits = data.get("result", {}).get(
        "points", []
    )

    server_seconds = data.get("time")

    server_ms = (
        float(server_seconds) * 1000
        if server_seconds is not None
        else None
    )

    overhead_ms = (
        elapsed_ms - server_ms
        if server_ms is not None
        else None
    )

    usage = data.get("usage") or {}

    # Qdrant supports different response
    # layouts across API versions.
    if isinstance(usage.get("hardware"), dict):
        usage = usage["hardware"]

    counters = {
        field: usage.get(field)
        for field in USAGE_FIELDS
    }

    violations = None

    if with_payload:
        violations = sum(
            1
            for hit in hits
            if (
                (
                    team is not None
                    and hit.get("payload", {}).get("team")
                    != team
                )
                or (
                    classification is not None
                    and hit.get("payload", {}).get(
                        "classification"
                    ) != classification
                )
            )
        )

    return {
        "client_ms": elapsed_ms,
        "server_ms": server_ms,
        "overhead_ms": overhead_ms,
        "usage": counters,
        "ids": [hit["id"] for hit in hits],
        "scores": [
            hit.get("score") for hit in hits
        ],
        "hits": hits,
        "violations": violations,
    }




def one_search(
    query_text,
    team=None,
    classification=None,
    top_k=10,
    exact=False,
):
    query_vector = encoder().encode(
        [query_text],
        normalize_embeddings=True,
    )[0].tolist()

    rows = []
    timings = []

    configs = [
        (SINGLE, "1 shard"),
        (THREE, "3 shards"),
    ]

    for collection, label in configs:
        result = run_query(
            None,
            collection,
            query_vector,
            team,
            classification,
            top_k,
            exact,
            True,
        )

        timing = {
            "configuration": label,
            "client_ms": result["client_ms"],
            "server_ms": result["server_ms"],
            "overhead_ms": result["overhead_ms"],
            "filter_violations": result["violations"],
            "hit_count": len(result["hits"]),
        }

        timing.update(result["usage"])
        timings.append(timing)

        for rank, hit in enumerate(
            result["hits"], 1
        ):
            payload = hit.get("payload") or {}

            rows.append({
                "configuration": label,
                "rank": rank,
                "point_id": hit["id"],
                "score": hit.get("score"),
                "team": TEAMS.get(
                    payload.get("team"),
                    payload.get("team"),
                ),
                "classification": LEVELS.get(
                    payload.get("classification"),
                    payload.get("classification"),
                ),
                "text": payload.get("text", ""),
            })

    return (
        pd.DataFrame(timings),
        pd.DataFrame(rows),
    )




def benchmark(
    team=None,
    classification=None,
    top_k=10,
    queries=100,
    warmup=10,
    seed=42,
    exact=False,
    progress=None,
):
    frame, vectors = prepare_embeddings()

    rng = random.Random(seed)

    choices = rng.sample(
        range(len(frame)),
        min(queries, len(frame)),
    )

    query_vectors = [
        vectors[i].tolist()
        for i in choices
    ]

    records = []
    overlap_records = []

    configs = [
        (SINGLE, "1 shard"),
        (THREE, "3 shards"),
    ]

    for index, query_vector in enumerate(
        query_vectors
    ):
        order = (
            configs
            if index % 2 == 0
            else list(reversed(configs))
        )

        query_results = {}

        for collection, label in order:
            result = run_query(
                None,
                collection,
                query_vector,
                team,
                classification,
                top_k,
                exact,
                False,
            )

            query_results[label] = result

            if index < warmup:
                continue

            record = {
                "query_index": index,
                "configuration": label,
                "client_ms": result["client_ms"],
                "server_ms": result["server_ms"],
                "overhead_ms": result["overhead_ms"],
                "hit_count": len(result["ids"]),
            }

            record.update(result["usage"])
            records.append(record)

        if index >= warmup:
            a = query_results["1 shard"]["ids"]
            b = query_results["3 shards"]["ids"]

            overlap = len(
                set(a) & set(b)
            ) / max(
                1, min(len(a), len(b), top_k)
            )

            overlap_records.append({
                "query_index": index,
                "topk_overlap": overlap,
            })

        if progress and index % 5 == 0:
            progress(
                f"Executed {index + 1}/{len(query_vectors)}"
            )

    raw = pd.DataFrame(records)

    if raw.empty:
        raise ValueError(
            "queries must be greater than warmup"
        )

    def percentile(series, q):
        return series.quantile(q)

    summary_rows = []

    for config, group in raw.groupby(
        "configuration"
    ):
        row = {
            "configuration": config,
            "samples": len(group),
        }

        for metric in [
            "client_ms",
            "server_ms",
            "overhead_ms",
        ]:
            values = group[metric].dropna()

            row[f"{metric}_mean"] = (
                values.mean()
                if not values.empty else None
            )

            row[f"{metric}_p50"] = (
                percentile(values, 0.50)
                if not values.empty else None
            )

            row[f"{metric}_p95"] = (
                percentile(values, 0.95)
                if not values.empty else None
            )

            row[f"{metric}_p99"] = (
                percentile(values, 0.99)
                if not values.empty else None
            )

            row[f"{metric}_min"] = (
                values.min()
                if not values.empty else None
            )

            row[f"{metric}_max"] = (
                values.max()
                if not values.empty else None
            )

        for field in USAGE_FIELDS:
            row[f"{field}_mean"] = (
                group[field].mean()
            )

            row[f"{field}_sum"] = (
                group[field].sum(min_count=1)
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    pairs = raw.pivot(
        index="query_index",
        columns="configuration",
        values=[
            "client_ms",
            "server_ms",
            "overhead_ms",
        ],
    ).dropna(
        subset=[
            ("client_ms", "1 shard"),
            ("client_ms", "3 shards"),
        ]
    )

    comparison = pd.DataFrame({
        "query_index": pairs.index,
    })

    for metric in [
        "client_ms",
        "server_ms",
        "overhead_ms",
    ]:
        comparison[f"{metric}_delta_3_minus_1"] = (
            pairs[(metric, "3 shards")]
            - pairs[(metric, "1 shard")]
        ).to_numpy()

    comparison = comparison.merge(
        pd.DataFrame(overlap_records),
        on="query_index",
        how="left",
    )

    return raw, summary, comparison
