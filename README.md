# bi-ui — AI-Native Cohort Builder

A stateless **Streamlit** front end for natural-language data exploration and
cohort materialization over **bi-compute**, the DuckDB **Quack** HTTP compute
engine (a Render Private Service).

```
You (plain English)
   │  Claude  ->  DuckDB SQL
   ▼
bi-ui (Streamlit, stateless)
   │  quack_query('quack:bi-compute:10000', sql)        # executed server-side
   ▼
bi-compute (DuckDB 1.5.3 + quack + lance)  ──►  R2 (LanceDB datasets)
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
- **Text-to-SQL.** The three core LanceDB datasets are hardcoded; their columns are read
  once via `DESCRIBE SELECT * FROM __lance_scan('s3://…')` and fed to Claude with a strict
  prompt requiring `__lance_scan('<path>')` access — never bare table names, and never the
  bare `FROM 's3://…'` replacement scan (the R2 dataset paths have no `.lance` suffix, so
  only `__lance_scan` reads them). The model returns a single DuckDB `SELECT`; the app
  enforces read-only via DuckDB's statement parser before executing.
- **Save Cohort.** Materializes the (un-limited) query as a Lance dataset on R2 with
  `COPY (<sql>) TO 's3://…/cohorts/<name>' (FORMAT lance)` on bi-compute — queryable
  afterward by the same `__lance_scan('<path>')`.

## Zero local storage

No dataframe or customer record is written to the container disk. Reads return to
memory only; every mutation is a server-side `COPY` to R2. Cohort names are
validated as safe SQL identifiers.

## Environment contract

| Var | Required | Description |
|-----|----------|-------------|
| `QUACK_TOKEN` | yes | Shared quack auth token; must equal bi-compute's. |
| `ANTHROPIC_API_KEY` | yes | For text-to-SQL (`shared-api-keys/prd`). |
| `COHORTS_R2_PREFIX` | for saving | R2 prefix for cohorts, e.g. `s3://<bucket>/cohorts`. |
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
