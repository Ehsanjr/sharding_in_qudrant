
"""Streamlit UI for Qdrant Custom Sharding Benchmark."""

import streamlit as st
import pandas as pd
import plotly.express as px

from dataset import TEAMS, LEVELS

from engine import (
    initialize,
    get_overview,
    one_search,
    benchmark,
    URL,
    MODEL_NAME,
)


# ============================================================
# Configuration
# ============================================================

st.set_page_config(
    page_title="Qdrant Custom Sharding Benchmark",
    page_icon="📊",
    layout="wide",
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

LATENCY_FIELDS = [
    "client_ms",
    "server_ms",
    "overhead_ms",
]

LATENCY_LABELS = {
    "client_ms": "Client Latency",
    "server_ms": "Qdrant Server Time",
    "overhead_ms": "Overhead",
}

CONFIG_COLORS = {
    "1 shard": "#2563eb",
    "3 shards": "#ea580c",
}


# ============================================================
# Helper Functions
# ============================================================

def format_dataframe(df, decimals=3):
    """Round numeric columns for display."""
    result = df.copy()

    numeric_cols = result.select_dtypes(
        include="number"
    ).columns

    result[numeric_cols] = result[
        numeric_cols
    ].round(decimals)

    return result


def csv_bytes(df):
    return df.to_csv(
        index=False
    ).encode("utf-8-sig")


def render_download(df, label, filename):
    st.download_button(
        label=label,
        data=csv_bytes(df),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def show_metric_cards(df, metric):
    if df.empty or metric not in df.columns:
        return

    cols = st.columns(2)

    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 2]:
            value = row.get(metric)

            st.metric(
                label=row["configuration"],
                value=(
                    f"{value:.3f} ms"
                    if pd.notna(value)
                    else "N/A"
                ),
            )


def render_usage_table(df):
    available = [
        c for c in USAGE_FIELDS
        if c in df.columns
    ]

    if not available:
        st.info(
            "Qdrant did not return resource "
            "usage counters."
        )
        return

    columns = ["configuration"] + available

    st.dataframe(
        format_dataframe(df[columns]),
        use_container_width=True,
        hide_index=True,
    )


def render_latency_distribution(raw, metric):
    if metric not in raw.columns:
        st.warning(f"Metric unavailable: {metric}")
        return

    data = raw.dropna(subset=[metric])

    if data.empty:
        st.info("No measurements for this metric.")
        return

    fig = px.box(
        data,
        x="configuration",
        y=metric,
        color="configuration",
        color_discrete_map=CONFIG_COLORS,
        points="outliers",
        title=(
            f"{LATENCY_LABELS.get(metric, metric)} "
            "Distribution"
        ),
        labels={
            "configuration": "Configuration",
            metric: "Latency (ms)",
        },
    )

    fig.update_layout(
        showlegend=False,
        height=420,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# Session State
# ============================================================

if "single_times" not in st.session_state:
    st.session_state.single_times = None

if "single_hits" not in st.session_state:
    st.session_state.single_hits = None

if "benchmark_raw" not in st.session_state:
    st.session_state.benchmark_raw = None

if "benchmark_summary" not in st.session_state:
    st.session_state.benchmark_summary = None

if "benchmark_pairs" not in st.session_state:
    st.session_state.benchmark_pairs = None

if "single_settings" not in st.session_state:
    st.session_state.single_settings = None

if "benchmark_settings" not in st.session_state:
    st.session_state.benchmark_settings = None


# ============================================================
# Header
# ============================================================

st.title("Qdrant Custom Sharding Benchmark")

st.caption(
    "Single Qdrant Server | "
    "10,000 Persian Chunks | "
    "1 Custom Shard vs 3 Team-Based Custom Shards"
)

st.info(
    "Goal: Compare retrieval latency between "
    "one shared custom shard and three "
    "team-specific custom shards."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("Environment")

    st.markdown("**Qdrant URL**")
    st.code(URL)

    st.markdown("**Embedding Model**")
    st.caption(MODEL_NAME)

    st.divider()

    st.subheader("Benchmark Scope")

    st.markdown(
        """
        - One Qdrant server
        - Two collections
        - 10,000 identical chunks
        - Identical embeddings
        - Custom Sharding only
        - Metadata filtering
        """
    )

    st.divider()

    st.subheader("Initialization")

    if st.button(
        "Initialize if missing",
        type="primary",
        use_container_width=True,
    ):

        progress_area = st.empty()

        try:
            with st.spinner(
                "Preparing embeddings and collections..."
            ):

                result = initialize(
                    progress=progress_area.info,
                    recreate=False,
                )

            st.success(
                f"Ready: {result['rows']:,} points "
                f"per collection | "
                f"{result['dimension']} dimensions"
            )

        except Exception as exc:
            st.exception(exc)

    with st.expander("Destructive Reset"):

        st.warning(
            "This operation deletes and rebuilds "
            "the two benchmark collections."
        )

        if st.button(
            "Rebuild benchmark collections",
            use_container_width=True,
        ):

            progress_area = st.empty()

            try:
                with st.spinner("Rebuilding..."):

                    result = initialize(
                        progress=progress_area.info,
                        recreate=True,
                    )

                st.success(
                    f"Rebuilt: {result['rows']:,} "
                    "points per collection"
                )

                st.session_state.single_times = None
                st.session_state.single_hits = None
                st.session_state.benchmark_raw = None
                st.session_state.benchmark_summary = None
                st.session_state.benchmark_pairs = None

            except Exception as exc:
                st.exception(exc)


# ============================================================
# Collection Information
# ============================================================

st.header("1. Collection Status")

try:
    overview = get_overview()

    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True,
    )

except Exception as exc:
    st.error(
        f"Qdrant unavailable: {exc}"
    )


# ============================================================
# Retrieval Settings
# ============================================================

st.header("2. Retrieval Settings")

col1, col2, col3, col4 = st.columns(4)

with col1:
    team_label = st.selectbox(
        "Team",
        ["All", *TEAMS.values()],
    )

with col2:
    classification_label = st.selectbox(
        "Classification",
        ["All", *LEVELS.values()],
    )

with col3:
    top_k = st.select_slider(
        "Top K",
        options=[1, 3, 5, 10, 20, 50],
        value=10,
    )

with col4:
    exact = st.toggle(
        "Exact Vector Search",
        value=False,
        help=(
            "Disable approximate vector search "
            "to compare exact retrieval behavior."
        ),
    )


team = next(
    (
        key
        for key, value in TEAMS.items()
        if value == team_label
    ),
    None,
)

classification = next(
    (
        key
        for key, value in LEVELS.items()
        if value == classification_label
    ),
    None,
)


if team:
    st.success(
        f"Team-based routing enabled: {team_label}"
    )
else:
    st.info(
        "All teams selected. The 3-shard "
        "collection will search across all "
        "team shards."
    )


# ============================================================
# Main Tabs
# ============================================================

tab1, tab2 = st.tabs([
    "Single Search",
    "Repeated Benchmark",
])


# ============================================================
# TAB 1: Single Search
# ============================================================

with tab1:

    st.header("3. Single Search Analysis")

    query_text = st.text_input(
        "Persian Search Query",
        "گزارش کنترل دسترسی کاربران و امنیت اطلاعات",
    )

    if st.button(
        "Search Both Collections",
        type="primary",
        key="single_search_button",
    ):

        try:

            with st.spinner(
                "Encoding query and searching Qdrant..."
            ):

                times, hits = one_search(
                    query_text,
                    team,
                    classification,
                    top_k,
                    exact,
                )

            st.session_state.single_times = times
            st.session_state.single_hits = hits

            st.session_state.single_settings = {
                "query": query_text,
                "team": team_label,
                "classification": classification_label,
                "top_k": top_k,
                "exact": exact,
            }

        except Exception as exc:
            st.exception(exc)

    if st.session_state.single_times is not None:

        times = st.session_state.single_times
        hits = st.session_state.single_hits

        st.divider()

        st.subheader("Search Configuration")

        st.json(
            st.session_state.single_settings,
            expanded=False,
        )

        # ----------------------------------------------------
        # Client Latency
        # ----------------------------------------------------

        st.subheader("Client Latency")

        show_metric_cards(
            times,
            "client_ms",
        )

        # ----------------------------------------------------
        # Server Latency
        # ----------------------------------------------------

        st.subheader("Qdrant Server Processing Time")

        show_metric_cards(
            times,
            "server_ms",
        )

        # ----------------------------------------------------
        # Overhead
        # ----------------------------------------------------

        st.subheader("Client / Server Overhead")

        show_metric_cards(
            times,
            "overhead_ms",
        )

        st.caption(
            "Overhead = Client Latency - Server Time. "
            "It includes transport, serialization, "
            "client-side processing and measurement "
            "differences. It is not pure network latency."
        )

        # ----------------------------------------------------
        # Detailed Latency Table
        # ----------------------------------------------------

        st.subheader("Detailed Latency Measurements")

        latency_cols = [
            "configuration",
            "client_ms",
            "server_ms",
            "overhead_ms",
            "hit_count",
            "filter_violations",
        ]

        available_cols = [
            col for col in latency_cols
            if col in times.columns
        ]

        st.dataframe(
            format_dataframe(times[available_cols]),
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Latency Breakdown Chart
        # ----------------------------------------------------

        st.subheader("Latency Breakdown")

        available_metrics = [
            c for c in LATENCY_FIELDS
            if c in times.columns
        ]

        latency_long = times.melt(
            id_vars="configuration",
            value_vars=available_metrics,
            var_name="metric",
            value_name="latency_ms",
        )

        latency_long["metric"] = latency_long[
            "metric"
        ].map(LATENCY_LABELS)

        fig = px.bar(
            latency_long,
            x="configuration",
            y="latency_ms",
            color="metric",
            barmode="group",
            title="Client vs Server vs Overhead",
            labels={
                "configuration": "Configuration",
                "latency_ms": "Latency (ms)",
                "metric": "Metric",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # CPU / I/O Counters
        # ----------------------------------------------------

        st.subheader("Qdrant CPU / I/O Counters")

        render_usage_table(times)

        st.caption(
            "These values are Qdrant resource counters, "
            "not CPU percentages or durations. "
            "Missing values are not interpreted as zero."
        )

        # ----------------------------------------------------
        # Retrieved Chunks
        # ----------------------------------------------------

        st.subheader("Retrieved Chunks")

        if hits is not None and not hits.empty:

            st.dataframe(
                hits,
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.warning(
                "No matching chunks returned."
            )

        # ----------------------------------------------------
        # Downloads
        # ----------------------------------------------------

        st.subheader("Export Single Search")

        dl1, dl2 = st.columns(2)

        with dl1:
            render_download(
                times,
                "Download Search Metrics",
                "single_search_metrics.csv",
            )

        with dl2:
            render_download(
                hits,
                "Download Retrieved Chunks",
                "single_search_results.csv",
            )

        st.caption(
            "Embedding generation and Streamlit "
            "rendering are excluded from measured "
            "retrieval latency."
        )


# ============================================================
# TAB 2: Repeated Benchmark
# ============================================================

with tab2:

    st.header("4. Repeated Benchmark")

    x, y, z = st.columns(3)

    with x:
        queries = st.slider(
            "Total Queries",
            min_value=20,
            max_value=500,
            value=100,
            step=10,
        )

    with y:
        warmup = st.slider(
            "Warm-up Queries",
            min_value=0,
            max_value=30,
            value=10,
        )

    with z:
        seed = st.number_input(
            "Query Seed",
            value=42,
            step=1,
        )

    st.caption(
        "Each query vector is searched in both "
        "collections. Execution order alternates "
        "to reduce order-related bias."
    )

    if warmup >= queries:
        st.warning(
            "Warm-up must be smaller than "
            "total queries."
        )

    run_disabled = warmup >= queries

    if st.button(
        "Run Controlled Benchmark",
        type="primary",
        disabled=run_disabled,
        key="benchmark_button",
    ):

        progress_area = st.empty()

        try:

            with st.spinner(
                "Running repeated retrieval benchmark..."
            ):

                raw, summary, pairs = benchmark(
                    team=team,
                    classification=classification,
                    top_k=top_k,
                    queries=queries,
                    warmup=warmup,
                    seed=int(seed),
                    exact=exact,
                    progress=progress_area.info,
                )

            st.session_state.benchmark_raw = raw
            st.session_state.benchmark_summary = summary
            st.session_state.benchmark_pairs = pairs

            st.session_state.benchmark_settings = {
                "team": team_label,
                "classification": classification_label,
                "top_k": top_k,
                "exact": exact,
                "queries": queries,
                "warmup": warmup,
                "seed": int(seed),
            }

            st.success(
                "Benchmark completed successfully."
            )

        except Exception as exc:
            st.exception(exc)

    # ========================================================
    # Benchmark Results
    # ========================================================

    if st.session_state.benchmark_raw is not None:

        raw = st.session_state.benchmark_raw
        summary = st.session_state.benchmark_summary
        pairs = st.session_state.benchmark_pairs

        st.divider()

        st.subheader("Benchmark Configuration")

        st.json(
            st.session_state.benchmark_settings,
            expanded=False,
        )

        # ----------------------------------------------------
        # Summary Table
        # ----------------------------------------------------

        st.subheader("Full Latency Summary")

        st.dataframe(
            format_dataframe(summary),
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Metric Selection
        # ----------------------------------------------------

        st.subheader("Latency Metric Analysis")

        selected_metric = st.selectbox(
            "Select Latency Metric",
            options=LATENCY_FIELDS,
            format_func=lambda value: LATENCY_LABELS[value],
            key="latency_metric_selector",
        )

        # ----------------------------------------------------
        # Distribution
        # ----------------------------------------------------

        render_latency_distribution(
            raw,
            selected_metric,
        )

        # ----------------------------------------------------
        # Percentiles
        # ----------------------------------------------------

        st.subheader("Latency Percentiles")

        percentile_columns = [
            "configuration",
            f"{selected_metric}_p50",
            f"{selected_metric}_p95",
            f"{selected_metric}_p99",
            f"{selected_metric}_min",
            f"{selected_metric}_max",
        ]

        available_percentiles = [
            c for c in percentile_columns
            if c in summary.columns
        ]

        st.dataframe(
            format_dataframe(
                summary[available_percentiles]
            ),
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Client / Server Average Comparison
        # ----------------------------------------------------

        st.subheader("Average Latency Breakdown")

        mean_columns = [
            "client_ms_mean",
            "server_ms_mean",
            "overhead_ms_mean",
        ]

        available_mean_columns = [
            c for c in mean_columns
            if c in summary.columns
        ]

        mean_data = summary[
            ["configuration"] + available_mean_columns
        ].melt(
            id_vars="configuration",
            var_name="metric",
            value_name="mean_ms",
        )

        mean_data["metric"] = mean_data[
            "metric"
        ].str.replace(
            "_mean", "", regex=False
        ).map(LATENCY_LABELS)

        fig_mean = px.bar(
            mean_data,
            x="configuration",
            y="mean_ms",
            color="metric",
            barmode="group",
            title="Average Client / Server / Overhead",
            labels={
                "mean_ms": "Average Latency (ms)",
                "configuration": "Configuration",
            },
        )

        st.plotly_chart(
            fig_mean,
            use_container_width=True,
        )

        # ----------------------------------------------------
        # Latency Over Query Order
        # ----------------------------------------------------

        st.subheader("Latency Over Query Order")

        chart_data = raw.dropna(
            subset=[selected_metric]
        )

        if not chart_data.empty:

            fig_line = px.line(
                chart_data,
                x="query_index",
                y=selected_metric,
                color="configuration",
                color_discrete_map=CONFIG_COLORS,
                title=(
                    f"{LATENCY_LABELS[selected_metric]} "
                    "Over Query Order"
                ),
                labels={
                    "query_index": "Query Index",
                    selected_metric: "Latency (ms)",
                },
            )

            st.plotly_chart(
                fig_line,
                use_container_width=True,
            )

        # ----------------------------------------------------
        # Paired Comparison
        # ----------------------------------------------------

        st.subheader("Paired Latency Comparison")

        delta_column = (
            f"{selected_metric}_delta_3_minus_1"
        )

        if delta_column in pairs.columns:

            delta_data = pairs.dropna(
                subset=[delta_column]
            )

            if not delta_data.empty:

                fig_delta = px.histogram(
                    delta_data,
                    x=delta_column,
                    nbins=30,
                    title=(
                        "Paired Delta: "
                        "3 Shards - 1 Shard"
                    ),
                    labels={
                        delta_column: "Delta (ms)"
                    },
                )

                fig_delta.add_vline(
                    x=0,
                    line_dash="dash",
                )

                st.plotly_chart(
                    fig_delta,
                    use_container_width=True,
                )

                mean_delta = delta_data[
                    delta_column
                ].mean()

                median_delta = delta_data[
                    delta_column
                ].median()

                c1, c2 = st.columns(2)

                c1.metric(
                    "Mean Delta",
                    f"{mean_delta:.3f} ms",
                )

                c2.metric(
                    "Median Delta",
                    f"{median_delta:.3f} ms",
                )

                if mean_delta < 0:
                    st.success(
                        "For this run, the 3-shard "
                        "configuration had lower average "
                        "latency on the selected metric."
                    )

                elif mean_delta > 0:
                    st.info(
                        "For this run, the 1-shard "
                        "configuration had lower average "
                        "latency on the selected metric."
                    )

                else:
                    st.info(
                        "Both configurations have equal "
                        "average latency for this run."
                    )

        st.caption(
            "Negative delta means 3 shards were faster. "
            "Positive delta means 1 shard was faster. "
            "One run alone does not establish "
            "statistical significance."
        )

        # ----------------------------------------------------
        # Top-K Overlap
        # ----------------------------------------------------

        st.subheader("Retrieval Result Agreement")

        if (
            "topk_overlap" in pairs.columns
            and not pairs.empty
        ):

            overlap = pairs[
                "topk_overlap"
            ].mean()

            st.metric(
                "Mean Top-K Overlap",
                f"{overlap:.2%}",
            )

            fig_overlap = px.histogram(
                pairs,
                x="topk_overlap",
                nbins=20,
                title="Top-K Overlap Distribution",
            )

            st.plotly_chart(
                fig_overlap,
                use_container_width=True,
            )

        st.caption(
            "Top-K overlap compares returned point IDs "
            "between configurations. It is not "
            "ground-truth Recall@K."
        )

        # ----------------------------------------------------
        # Qdrant CPU / I/O
        # ----------------------------------------------------

        st.subheader("Qdrant Resource Usage")

        usage_available = [
            field for field in USAGE_FIELDS
            if field in raw.columns
        ]

        if usage_available:

            usage_metric = st.selectbox(
                "Select Resource Counter",
                options=usage_available,
                key="usage_metric_selector",
            )

            resource_data = raw.dropna(
                subset=[usage_metric]
            )

            if not resource_data.empty:

                fig_usage = px.box(
                    resource_data,
                    x="configuration",
                    y=usage_metric,
                    color="configuration",
                    color_discrete_map=CONFIG_COLORS,
                    points="outliers",
                    title=(
                        f"{usage_metric} per Query"
                    ),
                )

                fig_usage.update_layout(
                    showlegend=False,
                )

                st.plotly_chart(
                    fig_usage,
                    use_container_width=True,
                )

            else:
                st.info(
                    "No data returned for this "
                    "resource counter."
                )

            # Resource averages
            st.subheader("Average Resource Counters")

            usage_mean_columns = [
                f"{field}_mean"
                for field in usage_available
                if f"{field}_mean" in summary.columns
            ]

            st.dataframe(
                format_dataframe(
                    summary[
                        ["configuration"]
                        + usage_mean_columns
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Resource totals
            st.subheader("Total Resource Counters")

            usage_sum_columns = [
                f"{field}_sum"
                for field in usage_available
                if f"{field}_sum" in summary.columns
            ]

            st.dataframe(
                format_dataframe(
                    summary[
                        ["configuration"]
                        + usage_sum_columns
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.info(
                "No resource counters were "
                "returned by Qdrant."
            )

        st.caption(
            "Resource counters are not elapsed time "
            "or CPU utilization percentages. "
            "Values depend on Qdrant instrumentation "
            "and cache behavior."
        )

        # ----------------------------------------------------
        # Raw Data
        # ----------------------------------------------------

        st.subheader("Raw Measurements")

        st.dataframe(
            format_dataframe(raw, decimals=4),
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Exports
        # ----------------------------------------------------

        st.subheader("Export Benchmark Results")

        d1, d2, d3 = st.columns(3)

        with d1:
            render_download(
                raw,
                "Download Raw Metrics",
                "qdrant_raw_metrics.csv",
            )

        with d2:
            render_download(
                summary,
                "Download Summary",
                "qdrant_summary.csv",
            )

        with d3:
            render_download(
                pairs,
                "Download Paired Comparison",
                "qdrant_pairs.csv",
            )


# ============================================================
# Interpretation Notes
# ============================================================

st.divider()

st.header("5. Interpretation Guide")

with st.expander(
    "How to interpret benchmark results",
    expanded=True,
):

    st.markdown(
        """
        **Client Latency**

        Total request/response time measured
        by the Python client.

        **Qdrant Server Time**

        Processing time reported directly
        by Qdrant REST API.

        **Overhead**

        Difference between client and server time.
        Includes transport, serialization and
        client-side overhead.

        **P50**

        Median retrieval latency.

        **P95**

        95th percentile of retrieval latency.

        **P99**

        99th percentile of retrieval latency.

        **CPU / I/O Counters**

        Qdrant resource usage counters.
        These are not CPU percentages or timings.

        **Top-K Overlap**

        Agreement between retrieved point IDs.
        It is not a measure of absolute Recall.

        **Important**

        This experiment compares:

        - 1 shared custom shard
        - 3 team-based custom shards

        All collections run on the same server.

        The experiment evaluates the combined
        effects of shard partitioning and
        team-based query routing.

        It does not isolate the number of shards
        from routing and data layout.
        """
    )
