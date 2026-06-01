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
- **Dataset catalog (Manifest Pattern).** Datasets are *not* discovered at boot. An offline
  job (`scripts/generate_manifest.py`, run by the data pipeline) recursively finds every Lance
  dataset under `active/`, extracts each schema, groups them by domain, and writes
  `active/catalog.json`. The UI reads that single small JSON instantly via bi-compute
  (`read_text`) and caches it (1 h) — no bucket globbing and no live `DESCRIBE` on the request
  path (a recursive `**` glob over `active/` never finishes at this scale). The full catalog
  schema is fed to the model.
- **Lance credentials.** `__lance_scan` authenticates to R2 via **AWS_\* env vars**
  (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` / `AWS_ENDPOINT`) on bi-compute —
  *not* the DuckDB s3 secret (which only covers httpfs glob/read_text). Both must be set on
  bi-compute or every Lance query fails with `Failed to get AWS credentials`.
- **Text-to-SQL.** The catalog schema is fed to Claude with a strict prompt requiring
  `__lance_scan('<path>')` access — never bare table names, and never the bare `FROM 's3://…'`
  replacement scan (the R2 dataset paths have no `.lance` suffix, so only `__lance_scan`
  reads them). The model returns a single DuckDB `SELECT`; the app enforces read-only via
  DuckDB's statement parser before executing.
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
| `LANCE_ACTIVE_PREFIX` | no | Active prefix; basis for the default catalog path. Defaults to `s3://data-sink/active`. |
| `CATALOG_PATH` | no | Catalog JSON path. Defaults to `<LANCE_ACTIVE_PREFIX>/catalog.json`. |
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

## Regenerating the catalog

Run whenever the Lance datasets change (the data pipeline calls this; needs R2 creds — the
script derives `AWS_*` from them for the lance `DESCRIBE`):

```bash
R2_ENDPOINT=… R2_ACCESS_KEY=… R2_SECRET_KEY=… R2_BUCKET=data-sink \
  python scripts/generate_manifest.py
```

Writes `s3://<bucket>/active/catalog.json` (one entry per Lance dataset, grouped by domain,
with each schema). The UI picks it up within the 1 h cache TTL or on redeploy.
