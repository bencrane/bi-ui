# bi-ui — AI-Native Audience Builder

A stateless **Streamlit** front end for natural-language data exploration and
audience materialization over **bi-compute**, the DuckDB **Quack** HTTP compute
engine (a Render Private Service).

```
You (plain English)
   │  Claude  ->  DuckDB SQL
   ▼
bi-ui (Streamlit, stateless)
   │  quack_query('quack:bi-compute:10000', sql)        # executed server-side
   ▼
bi-compute (DuckDB 1.5.3 + quack)  ──►  R2 (Parquet) + Lance
```

## How it works

- **Connection.** On startup the app opens an in-memory DuckDB, `INSTALL/LOAD quack`,
  creates a temporary `TYPE quack` secret from `QUACK_TOKEN`, and `ATTACH`es
  `quack:bi-compute:10000` as `data_sink`. The attach doubles as a fail-fast
  connectivity/auth gate.
- **Execution.** All SQL — reads, schema introspection, and writes — runs on
  bi-compute via `quack_query(uri, sql)`. The `ATTACH` alias itself is a streaming
  scan source and cannot serve joins or DDL in DuckDB 1.5.3, so `quack_query` (full
  server-side planner) is the execution path.
- **Text-to-SQL.** The live schema is introspected from bi-compute's
  `information_schema.columns` and fed to Claude with a strict read-only system
  prompt. The model returns a single DuckDB `SELECT`; the app enforces read-only
  before executing.
- **Save Audience.** Materializes the (un-limited) query straight to R2 with
  `COPY (<sql>) TO 's3://…/<name>/data.parquet'` on bi-compute, then registers a
  queryable `audiences.<name>` table backed by that object.

## Zero local storage

No dataframe or customer record is written to the container disk. Reads return to
memory only; every mutation is a server-side `COPY` to R2. Audience names are
validated as safe SQL identifiers.

## Environment contract

| Var | Required | Description |
|-----|----------|-------------|
| `QUACK_TOKEN` | yes | Shared quack auth token; must equal bi-compute's. |
| `ANTHROPIC_API_KEY` | yes | For text-to-SQL (`shared-api-keys/prd`). |
| `AUDIENCE_R2_PREFIX` | for saving | R2 prefix for audiences, e.g. `s3://<bucket>/audiences`. |
| `QUACK_URI` | no | Defaults to `quack:bi-compute:10000`. |
| `SQL_MODEL` | no | Defaults to `claude-sonnet-4-6`. |
| `PORT` | injected | Render sets it; Streamlit binds `0.0.0.0:$PORT`. |

## Deploy (Render)

Python web service built from this repo, **in the `ohio` region** (same as
bi-compute — private networking is regional). See `render.yaml`.

```bash
# local
pip install -r requirements.txt
streamlit run app.py
```
