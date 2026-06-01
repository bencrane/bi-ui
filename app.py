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

import os
import re

import duckdb
import pandas as pd
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
SQL_MODEL = os.environ.get("SQL_MODEL", "claude-sonnet-4-6")
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


# --- Lance dataset discovery: recursively find every dataset under ACTIVE_PREFIX ---
ACTIVE_PREFIX = os.environ.get("LANCE_ACTIVE_PREFIX", "s3://data-sink/active").rstrip("/")


def _domain_of(root: str) -> tuple[str, str]:
    """Return (domain, relative-name): domain is the first path segment under active/."""
    rel = root.split("/active/", 1)[-1] if "/active/" in root else root.rstrip("/").rsplit("/", 1)[-1]
    parts = rel.split("/")
    return (parts[0], rel) if len(parts) > 1 else ("(top level)", rel)


@st.cache_data(ttl=1800, show_spinner="Discovering Lance datasets in R2…")
def discover_datasets() -> tuple[str, dict[str, list[dict]]]:
    """Recursively discover every Lance dataset under ACTIVE_PREFIX and describe each.

    A Lance dataset root is identified by its `_versions/*.manifest` marker (the root is
    that path with the marker stripped). The glob and per-dataset DESCRIBE run server-side
    on bi-compute (httpfs + lance + R2). Cached for 30 min; a cold scan walks the whole
    active/ prefix, so first load is slow when there are many datasets.
    Returns (schema_text grouped by domain for the prompt, {domain: [{path, name, cols}]}).
    """
    con = get_connection()
    manifests = run_remote(
        con, f"SELECT file FROM glob('{_sql_str(ACTIVE_PREFIX)}/**/_versions/*.manifest')"
    ).fetchall()
    roots = sorted({f.rsplit("/_versions/", 1)[0] for (f,) in manifests})

    by_domain: dict[str, list[dict]] = {}
    for root in roots:
        domain, rel = _domain_of(root)
        try:
            cols = run_remote(
                con, f"DESCRIBE SELECT * FROM __lance_scan('{_sql_str(root)}')"
            ).fetchall()
            colstr = ", ".join(f"{c[0]} {c[1]}" for c in cols)
        except Exception as exc:
            colstr = f"(schema unavailable: {str(exc).splitlines()[0][:60]})"
        by_domain.setdefault(domain, []).append({"path": root, "name": rel, "cols": colstr})

    lines = []
    for domain in sorted(by_domain):
        lines.append(f"### domain: {domain}")
        for d in by_domain[domain]:
            lines.append(f"- {d['path']}\n    columns: {d['cols']}")
    schema_text = "\n".join(lines) if lines else f"(no Lance datasets found under {ACTIVE_PREFIX})"
    return schema_text, by_domain


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


# --- UI ----------------------------------------------------------------------
def main() -> None:
  st.set_page_config(page_title="bi-ui · Cohort Builder", layout="wide")
  st.title("🦆 Cohort Builder")
  st.caption("Natural-language exploration over bi-compute · cohorts materialize to R2")

  try:
    con = get_connection()
    schema_text, by_domain = discover_datasets()
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
        st.code("\n".join(d["name"] for d in by_domain[domain]), language=None)

  for turn in st.session_state.setdefault("history", []):
    with st.chat_message(turn["role"]):
      st.markdown(turn["content"])

  if question := st.chat_input("Describe the cohort in plain English…"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
      st.markdown(question)
    with st.chat_message("assistant"):
      try:
        sql = generate_sql(question, schema_text)
        assert_read_only(con, sql)
        df = run_remote(con, sql).df()
        st.session_state["last_sql"] = sql
        st.session_state["last_df"] = df
        st.code(sql, language="sql")
        st.markdown(f"**{len(df):,} rows**")
        st.dataframe(df, use_container_width=True)
        st.session_state.history.append(
          {"role": "assistant", "content": f"```sql\n{sql}\n```\nReturned {len(df):,} rows."}
        )
      except Exception as exc:
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
