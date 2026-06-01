"""
bi-ui — AI-native audience builder over the bi-compute DuckDB Quack engine.

A stateless Streamlit front end:
  • natural language  -> DuckDB SQL (Claude)
  • SQL executes *server-side* on bi-compute via the quack client/server protocol
  • audiences are materialized durably to R2 (COPY ... TO 's3://...')

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
# R2 destination for materialized audiences, e.g. s3://dex-curated/audiences
AUDIENCE_R2_PREFIX = os.environ.get("AUDIENCE_R2_PREFIX", "").rstrip("/")

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


# --- live schema, introspected server-side and fed to the model --------------
@st.cache_data(ttl=300, show_spinner="Loading live schema from bi-compute…")
def load_schema() -> tuple[str, list[str]]:
    con = get_connection()
    rows = run_remote(
        con,
        "SELECT table_schema, table_name, column_name, data_type "
        "FROM information_schema.columns "
        "WHERE table_schema NOT IN ('information_schema', 'pg_catalog', 'audiences') "
        "ORDER BY table_schema, table_name, ordinal_position",
    ).fetchall()

    tables: dict[str, list[str]] = {}
    for schema, table, column, dtype in rows:
        key = table if schema == "main" else f"{schema}.{table}"
        tables.setdefault(key, []).append(f"{column} {dtype}")

    lines = [f"- {name}({', '.join(cols)})" for name, cols in sorted(tables.items())]
    return "\n".join(lines) if lines else "(no tables visible)", sorted(tables)


SYSTEM_PROMPT = """You are a precise DuckDB SQL generator for an audience-building tool.

Your SQL executes on a remote DuckDB server (bi-compute), so reference every table
by the exact server-local name in the schema below — never prefix names with
'data_sink'. Emit DuckDB SQL dialect only.

LIVE SCHEMA (authoritative — use only these tables and columns):
{schema}

RULES:
1. Output exactly ONE statement and NOTHING else: no prose, no explanation, no
   markdown fences, no comments.
2. The statement MUST be read-only — a single SELECT, optionally led by WITH CTEs.
   Never emit INSERT/UPDATE/DELETE/CREATE/ATTACH/COPY/PRAGMA/SET/etc.
3. Use only tables and columns present in the schema above. If the request cannot
   be answered from this schema, output exactly: SELECT 'insufficient schema' AS error
4. Qualify columns when joining; prefer explicit column lists for audience definitions.
5. End with LIMIT 1000 unless the user explicitly asks for a full extract or an exact count.
6. Use standard DuckDB functions and syntax."""


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


def materialize_audience(con: duckdb.DuckDBPyConnection, name: str, select_sql: str) -> str:
    """Persist the audience to R2 (durable) and register a queryable catalog table.

    The mutation runs on bi-compute (which holds the R2 secret) via quack_query and
    writes straight to object storage — satisfying the 'all mutations pass through to
    R2' guardrail. The local container writes nothing.
    """
    if not _IDENT_RE.match(name):
        raise ValueError("Audience name must be lowercase letters/digits/underscores, ≤ 63 chars.")
    if not AUDIENCE_R2_PREFIX:
        raise RuntimeError("AUDIENCE_R2_PREFIX is not configured — cannot materialize to R2.")

    # Materialize the full population, not the previewed 1000.
    definition = _TRAILING_LIMIT_RE.sub("", select_sql).strip()
    target = f"{AUDIENCE_R2_PREFIX}/{name}/data.parquet"

    run_remote(con, f"COPY ({definition}) TO '{_sql_str(target)}' (FORMAT parquet)")
    run_remote(con, "CREATE SCHEMA IF NOT EXISTS audiences")
    run_remote(
        con,
        f'CREATE OR REPLACE TABLE audiences."{name}" AS '
        f"SELECT * FROM read_parquet('{_sql_str(target)}')",
    )
    return target


# --- UI ----------------------------------------------------------------------
def main() -> None:
  st.set_page_config(page_title="bi-ui · Audience Builder", layout="wide")
  st.title("🦆 Audience Builder")
  st.caption("Natural-language exploration over bi-compute · audiences materialize to R2")

  try:
    con = get_connection()
    schema_text, table_names = load_schema()
  except Exception as exc:  # surface the live hop clearly instead of a blank page
    st.error(f"Could not initialize the bi-compute connection: {exc}")
    st.stop()

  with st.sidebar:
    st.subheader("Connection")
    st.write(f"**Endpoint:** `{QUACK_URI}` ({'plain HTTP' if QUACK_DISABLE_SSL else 'TLS'})")
    st.write(f"**Model:** `{SQL_MODEL}`")
    st.write(f"**R2 sink:** `{AUDIENCE_R2_PREFIX or '⚠ not configured'}`")
    st.subheader(f"Tables ({len(table_names)})")
    st.code("\n".join(table_names) or "(none)", language=None)

  for turn in st.session_state.setdefault("history", []):
    with st.chat_message(turn["role"]):
      st.markdown(turn["content"])

  if question := st.chat_input("Describe the audience in plain English…"):
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

  # --- Save Audience: only when there is a successful result to materialize ---
  if isinstance(st.session_state.get("last_df"), pd.DataFrame):
    st.divider()
    left, right = st.columns([3, 1])
    with left:
      audience_name = st.text_input(
        "Audience name",
        placeholder="e.g. fmcsa_high_intent_q3",
        help="Lowercase letters, digits, underscores. Becomes audiences.<name> and the R2 object key.",
      )
    with right:
      st.write("")
      st.write("")
      if st.button("💾 Save Audience", type="primary", use_container_width=True):
        try:
          target = materialize_audience(con, audience_name.strip(), st.session_state["last_sql"])
          st.success(f"Materialized → `{target}` and registered `audiences.{audience_name.strip()}`.")
        except Exception as exc:
          st.error(f"Save failed: {exc}")


if __name__ == "__main__":
  main()
