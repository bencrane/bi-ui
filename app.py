"""
bi-ui — AI-native cohort builder over the bi-compute DuckDB Quack engine.

A stateless Streamlit front end:
  • natural language  -> DuckDB SQL (Claude)
  • SQL executes *server-side* on bi-compute via the quack client/server protocol
  • cohorts are materialized durably to R2 (COPY ... TO 's3://...')

Zero local storage: no dataframe and no customer record is ever written to this
container's disk. Every read and every mutation is executed on bi-compute through
the quack endpoint; result sets live only in process memory for the session.

Why everything routes through quack_query() instead of the ATTACH alias:
  The quack ATTACH is a streaming-scan source. DuckDB 1.5.3 cannot run multiple
  streaming scans (any JOIN) or CTAS/INSERT against it ("Multiple streaming scans
  ... not currently supported"). quack_query(uri, sql) executes the statement on
  the remote server with a full local planner, so joins, DDL and COPY all work.
  The ATTACH is kept as the directive-mandated mount and as a fail-fast
  connectivity/auth gate at startup.
"""
from __future__ import annotations

import json
import os
import re
from typing import NamedTuple

import altair as alt
import duckdb
import pandas as pd
import pydeck as pdk
import streamlit as st
from anthropic import Anthropic

# --- configuration: everything via env; no secrets on disk -------------------
QUACK_URI = os.environ.get("QUACK_URI", "quack:bi-compute:10000")
QUACK_TOKEN = os.environ.get("QUACK_TOKEN", "")
# bi-compute serves plain HTTP (quack_serve disable_ssl := true). The quack client
# defaults to SSL for hostnames (only loopback IPs default to plain), so we must
# disable SSL explicitly on both ATTACH and quack_query or every call fails with
# "SSL connect error". Flip to false only if bi-compute is fronted by TLS.
QUACK_DISABLE_SSL = os.environ.get("QUACK_DISABLE_SSL", "true").lower() == "true"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SQL_MODEL = os.environ.get("SQL_MODEL", "claude-opus-4-7")
# R2 destination for materialized cohorts, e.g. s3://data-sink/active/cohorts
COHORTS_R2_PREFIX = os.environ.get("COHORTS_R2_PREFIX", "").rstrip("/")

ATTACH_ALIAS = "data_sink"
_DISABLE_SSL_SQL = "true" if QUACK_DISABLE_SSL else "false"
_IDENT_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_TRAILING_LIMIT_RE = re.compile(r"\s+limit\s+\d+(\s+offset\s+\d+)?\s*;?\s*$", re.IGNORECASE)


def _sql_str(value: str) -> str:
    """Escape a value for embedding as a single-quoted SQL literal."""
    return value.replace("'", "''")


# --- connection: one in-memory client; SECRET + ATTACH as a fast health gate -
@st.cache_resource(show_spinner="Connecting to bi-compute…")
def get_connection() -> duckdb.DuckDBPyConnection:
    if not QUACK_TOKEN:
        raise RuntimeError("QUACK_TOKEN is not set in the environment.")
    con = duckdb.connect()  # in-memory — nothing touches local disk
    con.execute("INSTALL quack; LOAD quack;")
    # Temporary (in-memory) secret. Authorizes BOTH attach and quack_query().
    con.execute(f"CREATE SECRET (TYPE quack, TOKEN '{_sql_str(QUACK_TOKEN)}')")
    # Directive-mandated mount; also fails fast on bad host/token before the UI loads.
    # DISABLE_SSL must match the server (plain HTTP) or the attach throws "SSL connect error".
    con.execute(f"ATTACH '{_sql_str(QUACK_URI)}' AS {ATTACH_ALIAS} (TYPE quack, DISABLE_SSL {_DISABLE_SSL_SQL})")
    return con


def run_remote(con: duckdb.DuckDBPyConnection, sql: str):
    """Execute `sql` on bi-compute via quack_query (server-side). Returns a relation."""
    return con.execute(
        f"SELECT * FROM quack_query(?, ?, disable_ssl := {_DISABLE_SSL_SQL})", [QUACK_URI, sql]
    )


# --- Dataset catalog: read the pre-built manifest (instant; no globbing/DESCRIBE at boot) ---
# Built offline by scripts/generate_manifest.py and written to <active>/catalog.json. The UI
# reads that one small JSON through bi-compute (httpfs read_text), so boot never scans R2.
ACTIVE_PREFIX = os.environ.get("LANCE_ACTIVE_PREFIX", "s3://data-sink/active").rstrip("/")
CATALOG_PATH = os.environ.get("CATALOG_PATH", f"{ACTIVE_PREFIX}/catalog.json")


@st.cache_data(ttl=3600, show_spinner="Loading dataset catalog…")
def load_catalog() -> tuple[str, dict[str, list[dict]]]:
    """Read the pre-compiled catalog.json and build the LLM schema text + per-domain map for
    the sidebar. One small read through bi-compute — no bucket globbing, no live DESCRIBE on
    the request path (that work lives in scripts/generate_manifest.py, run by the pipeline).
    """
    con = get_connection()
    rows = run_remote(con, f"SELECT content FROM read_text('{_sql_str(CATALOG_PATH)}')").fetchall()
    catalog = json.loads(rows[0][0])
    domains: dict[str, list[dict]] = catalog.get("domains", {})

    lines = []
    for domain in sorted(domains):
        lines.append(f"### domain: {domain}")
        for d in domains[domain]:
            cols = ", ".join(f"{c} {t}" for c, t in d.get("schema", {}).items())
            lines.append(f"- {d['path']}\n    columns: {cols}")
    schema_text = "\n".join(lines) if lines else "(catalog is empty)"
    return schema_text, domains


SYSTEM_PROMPT = """You are a precise DuckDB SQL generator for a cohort-building tool. All data
lives in LanceDB datasets on R2; your SQL runs server-side on bi-compute (DuckDB + lance extension).

CRITICAL — how to read a dataset (DuckDB lance extension):
- Read a dataset ONLY via the lance scan function on its quoted S3 path:
      SELECT ... FROM __lance_scan('s3://data-sink/active/<dataset>') WHERE ...
- Join datasets by giving each scan an alias:
      FROM __lance_scan('s3://.../a') a JOIN __lance_scan('s3://.../b') b ON a.key = b.key
- NEVER use a bare table name (FROM bridge_sam_fmcsa_domain is INVALID — there are no named tables).
- NEVER use FROM 's3://...' directly, lance_scan(), read_lance(), or read_parquet(); only __lance_scan() reads these paths.
- DuckDB SQL dialect only.

AVAILABLE DATASETS — discovered live from R2, grouped by domain (use only these paths and columns):
{schema}

RULES:
1. Output exactly ONE statement and NOTHING else: no prose, no markdown fences, no comments.
2. Read-only: a single SELECT, optionally led by WITH CTEs. Never INSERT/UPDATE/DELETE/CREATE/COPY/ATTACH/PRAGMA/SET.
3. Use only the dataset paths and columns listed above. If the request cannot be satisfied,
   output exactly: SELECT 'insufficient schema' AS error
4. Qualify columns when joining; prefer explicit column lists for cohort definitions.
5. End with LIMIT 1000 unless the user explicitly asks for a full extract or an exact count."""


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip().rstrip(";").strip()


def generate_sql(question: str, schema_text: str) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=SQL_MODEL,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT.format(schema=schema_text),
                "cache_control": {"type": "ephemeral"},  # schema is reused every turn
            }
        ],
        messages=[{"role": "user", "content": question}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return strip_fences(text)


def assert_read_only(con: duckdb.DuckDBPyConnection, sql: str) -> None:
    """Defense in depth: exploration must never mutate, regardless of model output.

    Uses DuckDB's own parser to classify the statement — robust against keywords
    that appear inside string literals or identifiers, which a regex is not.
    """
    try:
        statements = con.extract_statements(sql)
    except Exception as exc:
        raise ValueError(f"Could not parse generated SQL: {exc}")
    if len(statements) != 1:
        raise ValueError("Exactly one statement may be run.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise ValueError(f"Only read-only SELECT queries are allowed (parsed as {statements[0].type}).")


def materialize_cohort(con: duckdb.DuckDBPyConnection, name: str, select_sql: str) -> str:
    """Persist the cohort as a Lance dataset on R2 (matches the data architecture).

    The COPY runs on bi-compute (which holds the R2 secret) via quack_query and writes
    straight to object storage — the local container writes nothing. The result is a
    Lance dataset at <prefix>/<name>, queryable by the same __lance_scan('<path>').
    """
    if not _IDENT_RE.match(name):
        raise ValueError("Cohort name must be lowercase letters/digits/underscores, ≤ 63 chars.")
    if not COHORTS_R2_PREFIX:
        raise RuntimeError("COHORTS_R2_PREFIX is not configured — cannot materialize to R2.")

    # Materialize the full population, not the previewed 1000.
    definition = _TRAILING_LIMIT_RE.sub("", select_sql).strip()
    target = f"{COHORTS_R2_PREFIX}/{name}"

    run_remote(con, f"COPY ({definition}) TO '{_sql_str(target)}' (FORMAT lance)")
    return target


# --- Result visualization: route a result set to metrics + theme-matched charts ---
# Detection is name/dtype-driven and computed once per result (classify -> Layout), so a
# single column is never charted two ways. Every Altair/PyDeck surface uses Streamlit's
# native theme (theme="streamlit") so fonts, gridlines and backgrounds track the dark
# palette pinned in .streamlit/config.toml.
ACCENT = "#38bdf8"        # sky-blue terminal accent — shared by chart marks and the map layer
STREAM_BATCH_ROWS = 4096  # client-side Arrow batch size for progressive paint

_LAT_CANDS = ("latitude", "lat", "lat_deg", "latitude_deg")
_LON_CANDS = ("longitude", "lon", "lng", "long", "lon_deg", "longitude_deg")
_TIME_HINTS = ("action_date", "signed_date", "date_signed", "period_of_performance_start_date")
_CAT_HINTS = (
    "naics_code", "naics", "agency", "sub_agency", "awarding_agency", "funding_agency",
    "department", "sector", "industry", "set_aside", "product_or_service_code", "psc",
    "state", "status", "type", "category", "country", "city",
)
_MONEY_RE = re.compile(
    r"(amount|amt|value|oblig|dollar|total|award|ceiling|price|revenue|spend|cost|fund|balance)", re.I
)
_CODEISH_RE = re.compile(r"(code|_id$|^id$|zip|fips|naics|psc|cage|duns|uei|rank|year|^yr$|^fy)", re.I)
_ENTITY_RE = re.compile(
    r"(uei|duns|recipient|vendor|company|legal_business_name|dot_number|cage|ein|awardee|name)", re.I
)
_STATE_RE = re.compile(r"(^state$|_state$|state_code|state_abbr)", re.I)


class Layout(NamedTuple):
    """The chartable roles detected in a result set (any field may be None)."""
    lat: str | None
    lon: str | None
    state: str | None
    time: str | None
    value: str | None
    entity: str | None
    category: str | None


def _first_present(df: pd.DataFrame, cands: tuple[str, ...]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in low:
            return low[k]
    return None


def _time_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    for c in df.columns:
        lc = c.lower()
        if lc in _TIME_HINTS or lc.endswith("_date") or lc.endswith("_at") or lc == "date":
            return c
    return None


def _value_column(df: pd.DataFrame, exclude: set[str | None]) -> str | None:
    """First numeric measure: money-named columns win; codes/ids/years are never measures."""
    num = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude and not _CODEISH_RE.search(c)
    ]
    money = [c for c in num if _MONEY_RE.search(c)]
    for c in (money or num):
        if df[c].notna().any():
            return c
    return None


def _entity_column(df: pd.DataFrame, exclude: set[str | None]) -> str | None:
    for c in df.columns:
        if c not in exclude and _ENTITY_RE.search(c):
            return c
    return None


def _category_column(df: pd.DataFrame, exclude: set[str | None]) -> str | None:
    for h in _CAT_HINTS:
        col = _first_present(df, (h,))
        if col and col not in exclude and df[col].nunique(dropna=True) > 1:
            return col
    best, best_n = None, None
    for c in df.columns:
        if c in exclude:
            continue
        if not (pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)):
            continue
        n = df[c].nunique(dropna=True)
        if 1 < n <= 50 and (best_n is None or n < best_n):
            best, best_n = c, n
    return best


def classify(df: pd.DataFrame) -> Layout:
    """Detect the chartable role of each column once, so routing is deterministic."""
    lat = _first_present(df, _LAT_CANDS)
    lon = _first_present(df, _LON_CANDS)
    state = next((c for c in df.columns if _STATE_RE.search(c)), None)
    time = _time_column(df)
    value = _value_column(df, {lat, lon, time})
    entity = _entity_column(df, {lat, lon, time, value})
    # lat/lon are represented by the map; state remains free to drive a ranked bar.
    category = _category_column(df, {c for c in (lat, lon, time, value, entity) if c})
    return Layout(lat, lon, state, time, value, entity, category)


def _looks_monetary(name: str | None) -> bool:
    return bool(_MONEY_RE.search(name or ""))


def _human_num(x, money: bool = False) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    sign, unit, a = ("-" if x < 0 else ""), ("$" if money else ""), abs(x)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if a >= div:
            return f"{sign}{unit}{a / div:,.1f}{suf}"
    return f"{sign}{unit}{a:,.0f}"


def _bucket_freq(s: pd.Series) -> str:
    days = getattr(s.max() - s.min(), "days", 0) or 0
    if days <= 120:
        return "D"
    if days <= 1100:
        return "W"
    if days <= 4000:
        return "ME"
    return "YE"


def render_metrics(df: pd.DataFrame, layout: Layout) -> None:
    """KPI strip: row/column counts plus a detected measure sum and a distinct-entity count."""
    tiles = [("Rows", f"{len(df):,}"), ("Columns", f"{df.shape[1]:,}")]
    if layout.value:
        tiles.append(
            (f"Σ {layout.value}", _human_num(df[layout.value].sum(), money=_looks_monetary(layout.value)))
        )
    distinct_col = layout.entity or layout.category
    if distinct_col and len(tiles) < 4:
        tiles.append((f"Distinct {distinct_col}", f"{df[distinct_col].nunique():,}"))
    for col, (label, val) in zip(st.columns(len(tiles)), tiles):
        col.metric(label, val)


def _geo_deck(df: pd.DataFrame, layout: Layout):
    """PyDeck scatter (or heatmap when dense) over lat/lon — token-free, dark base canvas."""
    if not (layout.lat and layout.lon):
        return None
    pts = pd.DataFrame({
        "lat": pd.to_numeric(df[layout.lat], errors="coerce"),
        "lon": pd.to_numeric(df[layout.lon], errors="coerce"),
    }).dropna()
    pts = pts[pts["lat"].between(-90, 90) & pts["lon"].between(-180, 180)]
    if pts.empty:
        return None
    view = pdk.ViewState(
        latitude=float(pts["lat"].mean()), longitude=float(pts["lon"].mean()),
        zoom=3 if len(pts) > 1 else 8, pitch=0,
    )
    if len(pts) > 5000:
        layer = pdk.Layer("HeatmapLayer", data=pts, get_position="[lon, lat]", aggregation="SUM")
    else:
        layer = pdk.Layer(
            "ScatterplotLayer", data=pts, get_position="[lon, lat]",
            get_radius=14000, radius_min_pixels=2, radius_max_pixels=24,
            get_fill_color=[56, 189, 248, 150], pickable=True,
        )
    return pdk.Deck(
        layers=[layer], initial_view_state=view, map_style=None,
        tooltip={"text": "{lat}, {lon}"},
    )


def _time_chart(df: pd.DataFrame, layout: Layout):
    """Theme-matched area chart: a detected measure summed (else row count) over the time axis."""
    if not layout.time:
        return None
    work = pd.DataFrame({"t": pd.to_datetime(df[layout.time], errors="coerce")})
    if layout.value:
        work["v"] = pd.to_numeric(df[layout.value], errors="coerce")
    work = work.dropna(subset=["t"])
    if work.empty:
        return None
    freq = _bucket_freq(work["t"])
    if layout.value:
        g = work.dropna(subset=["v"]).groupby(pd.Grouper(key="t", freq=freq))["v"].sum().reset_index()
        ycol, ytitle = "v", f"Σ {layout.value}"
    else:
        g = work.groupby(pd.Grouper(key="t", freq=freq)).size().reset_index(name="rows")
        ycol, ytitle = "rows", "rows"
    if g.empty:
        return None
    gradient = alt.Gradient(
        gradient="linear",
        stops=[alt.GradientStop(color=ACCENT, offset=0), alt.GradientStop(color="#0b1220", offset=1)],
        x1=1, x2=1, y1=0, y2=1,
    )
    return (
        alt.Chart(g, title=f"{ytitle} over {layout.time}")
        .mark_area(line={"color": ACCENT}, color=gradient, opacity=0.7)
        .encode(
            x=alt.X("t:T", title=layout.time),
            y=alt.Y(f"{ycol}:Q", title=ytitle),
            tooltip=[alt.Tooltip("t:T", title=layout.time),
                     alt.Tooltip(f"{ycol}:Q", title=ytitle, format=",.0f")],
        )
        .properties(height=280)
    )


def _category_chart(df: pd.DataFrame, layout: Layout):
    """Theme-matched horizontal ranked bar: top-15 categories by measure sum (else row count)."""
    c = layout.category
    if not c:
        return None
    if layout.value:
        g = (
            df.assign(_v=pd.to_numeric(df[layout.value], errors="coerce"))
            .groupby(c)["_v"].sum().sort_values(ascending=False).head(15).reset_index()
        )
        xcol, xtitle = "_v", f"Σ {layout.value}"
    else:
        g = df[c].value_counts(dropna=True).head(15).rename_axis(c).reset_index(name="rows")
        xcol, xtitle = "rows", "rows"
    if g.empty:
        return None
    g[c] = g[c].astype(str)
    return (
        alt.Chart(g, title=f"top {len(g)} · {c}")
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X(f"{xcol}:Q", title=xtitle),
            y=alt.Y(f"{c}:N", sort="-x", title=c),
            tooltip=[alt.Tooltip(f"{c}:N"), alt.Tooltip(f"{xcol}:Q", title=xtitle, format=",.0f")],
        )
        .properties(height=max(220, 26 * len(g)))
    )


def render_charts(df: pd.DataFrame, layout: Layout) -> None:
    """Lay detected panels into a responsive grid: map full-width, then time | category."""
    deck = _geo_deck(df, layout)
    if deck is not None:
        st.caption("📍 geographic distribution")
        st.pydeck_chart(deck)
    panels = [p for p in (_time_chart(df, layout), _category_chart(df, layout)) if p is not None]
    if len(panels) == 2:
        for col, ch in zip(st.columns(2), panels):
            col.altair_chart(ch, use_container_width=True, theme="streamlit")
    elif panels:
        st.altair_chart(panels[0], use_container_width=True, theme="streamlit")
    if deck is None and not panels:
        st.info("No geographic, temporal, or categorical dimension detected — see the rows below.")


def stream_result(con, sql: str, *, status, metrics_slot, chart_slot) -> pd.DataFrame:
    """Pull the server-side result as Arrow batches, painting live counts + an interim chart.

    quack_query returns the full result set (no server-side scan counter exists to surface
    honestly), but DuckDB hands it back in batches — so the page paints progressively and
    never feels locked. "Rows streamed" is the true running count of rows pulled client-side.
    """
    reader = run_remote(con, sql).fetch_record_batch(STREAM_BATCH_ROWS)
    ncols = len(reader.schema)
    status.update(label="Probing bi-compute · streaming LanceDB result…")
    frames: list[pd.DataFrame] = []
    rows = 0
    for i, batch in enumerate(reader):
        frames.append(batch.to_pandas())
        rows += batch.num_rows
        with metrics_slot.container():
            a, b = st.columns(2)
            a.metric("Rows streamed", f"{rows:,}")
            b.metric("Columns", f"{ncols:,}")
        if i % 5 == 0:  # interim time-series paint; final dashboard replaces this slot
            interim = pd.concat(frames, ignore_index=True)
            ch = _time_chart(interim, classify(interim))
            if ch is not None:
                chart_slot.altair_chart(ch, use_container_width=True, theme="streamlit")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return reader.schema.empty_table().to_pandas()


def render_result(sql: str, df: pd.DataFrame, layout: Layout, *, metrics_slot, chart_slot, table_slot) -> None:
    """Paint the finalized dashboard into the pre-allocated slots; raw rows + SQL stay one click away."""
    with metrics_slot.container():
        render_metrics(df, layout)
    with chart_slot.container():
        if len(df):
            render_charts(df, layout)
        else:
            st.info("Query returned 0 rows.")
    with table_slot.container():
        with st.expander(f"🔎 generated SQL · {len(df):,} rows", expanded=False):
            st.code(sql, language="sql")
            st.dataframe(df, use_container_width=True)


# --- UI ----------------------------------------------------------------------
def main() -> None:
  st.set_page_config(page_title="bi-ui · Cohort Builder", layout="wide")
  st.title("🦆 Cohort Builder")
  st.caption("Natural-language exploration over bi-compute · cohorts materialize to R2")

  try:
    con = get_connection()
    schema_text, by_domain = load_catalog()
  except Exception as exc:  # surface the live hop clearly instead of a blank page
    st.error(f"Could not initialize the bi-compute connection: {exc}")
    st.stop()

  with st.sidebar:
    st.subheader("Connection")
    st.write(f"**Endpoint:** `{QUACK_URI}` ({'plain HTTP' if QUACK_DISABLE_SSL else 'TLS'})")
    st.write(f"**Model:** `{SQL_MODEL}`")
    st.write(f"**R2 sink:** `{COHORTS_R2_PREFIX or '⚠ not configured'}`")
    total = sum(len(v) for v in by_domain.values())
    st.subheader(f"Lance datasets ({total})")
    for domain in sorted(by_domain):
      with st.expander(f"{domain} ({len(by_domain[domain])})"):
        st.code("\n".join(d["dataset_name"] for d in by_domain[domain]), language=None)

  for turn in st.session_state.setdefault("history", []):
    with st.chat_message(turn["role"]):
      st.markdown(turn["content"])

  if question := st.chat_input("Describe the cohort in plain English…"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
      st.markdown(question)
    with st.chat_message("assistant"):
      status_slot = st.empty()
      sql_slot = st.empty()
      metrics_slot = st.empty()
      chart_slot = st.empty()
      table_slot = st.empty()
      try:
        with status_slot.status("Generating DuckDB SQL…", expanded=True) as status:
          sql = generate_sql(question, schema_text)
          sql_slot.code(sql, language="sql")
          assert_read_only(con, sql)
          df = stream_result(con, sql, status=status, metrics_slot=metrics_slot, chart_slot=chart_slot)
          status.update(label=f"Done · {len(df):,} rows", state="complete")
        status_slot.empty()
        sql_slot.empty()  # SQL moves into the result expander; keep the dashboard clean
        st.session_state["last_sql"] = sql
        st.session_state["last_df"] = df
        render_result(
          sql, df, classify(df),
          metrics_slot=metrics_slot, chart_slot=chart_slot, table_slot=table_slot,
        )
        st.session_state.history.append(
          {"role": "assistant", "content": f"```sql\n{sql}\n```\nReturned {len(df):,} rows."}
        )
      except Exception as exc:
        status_slot.empty()
        st.error(f"{exc}")
        st.session_state.history.append({"role": "assistant", "content": f"⚠ {exc}"})

  # --- Save Cohort: only when there is a successful result to materialize ---
  if isinstance(st.session_state.get("last_df"), pd.DataFrame):
    st.divider()
    left, right = st.columns([3, 1])
    with left:
      cohort_name = st.text_input(
        "Cohort name",
        placeholder="e.g. fmcsa_high_intent_q3",
        help="Lowercase letters/digits/underscores — the Lance dataset name written under the cohorts R2 prefix.",
      )
    with right:
      st.write("")
      st.write("")
      if st.button("💾 Save Cohort", type="primary", use_container_width=True):
        try:
          target = materialize_cohort(con, cohort_name.strip(), st.session_state["last_sql"])
          st.success(f"Materialized Lance dataset → `{target}` · query with `__lance_scan('{target}')`")
        except Exception as exc:
          st.error(f"Save failed: {exc}")


if __name__ == "__main__":
  main()
