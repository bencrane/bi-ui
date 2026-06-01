#!/usr/bin/env python3
"""
generate_manifest.py — build the bi-ui dataset catalog (the "Manifest Pattern").

Run by the data pipeline whenever Lance datasets change. It:
  1. recursively discovers every Lance dataset under the active/ prefix (bounded-depth
     glob on the `_versions/*.manifest` marker — never a `**` scan, which would walk
     every Lance fragment and never finish),
  2. extracts each dataset's schema via `DESCRIBE … __lance_scan(<path>)`,
  3. groups datasets by their first path segment (domain),
  4. writes the result to s3://<bucket>/active/catalog.json.

The UI then reads that single small JSON instantly at boot — no globbing, no live
DESCRIBE loops on the request path.

IMPORTANT — credentials. The DuckDB **lance** reader authenticates to S3/R2 via the
lance Rust object_store, which reads AWS_* environment variables, NOT DuckDB's
`CREATE SECRET`. httpfs (glob/read_text) uses the DuckDB secret. So we set BOTH:
a DuckDB s3 secret (for the glob) and AWS_* env (for __lance_scan).

Env contract (R2):
  R2_ENDPOINT       R2 S3 endpoint (host or https://host)
  R2_ACCESS_KEY     R2 access key id
  R2_SECRET_KEY     R2 secret access key
  R2_BUCKET         bucket name (default: data-sink)
  R2_ACTIVE_PREFIX  scan root  (default: s3://<bucket>/active)
  CATALOG_KEY       output key (default: active/catalog.json)
  MAX_DEPTH         dataset nesting depth to scan (default: 5)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import boto3
import duckdb

BUCKET = os.environ.get("R2_BUCKET", "data-sink")
ENDPOINT_RAW = os.environ["R2_ENDPOINT"]
HOST = ENDPOINT_RAW.removeprefix("https://").removeprefix("http://").rstrip("/")
ENDPOINT_URL = f"https://{HOST}"
KEY_ID = os.environ["R2_ACCESS_KEY"]
SECRET = os.environ["R2_SECRET_KEY"]
ACTIVE_PREFIX = os.environ.get("R2_ACTIVE_PREFIX", f"s3://{BUCKET}/active").rstrip("/")
CATALOG_KEY = os.environ.get("CATALOG_KEY", "active/catalog.json")
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "5"))


def _domain_of(root: str) -> tuple[str, str]:
    """(domain, relative-name). Foldered datasets group by their folder; flat datasets
    group by the leading underscore token (edgar_form_4 -> edgar, cms_* -> cms, …)."""
    rel = root.split("/active/", 1)[-1] if "/active/" in root else root.rstrip("/").rsplit("/", 1)[-1]
    parts = rel.split("/")
    if len(parts) > 1:
        return parts[0], rel
    name = parts[0]
    return (name.split("_", 1)[0] if "_" in name else name), rel


def _connect() -> duckdb.DuckDBPyConnection:
    # lance reader needs AWS_* env; set them from the R2 creds before any __lance_scan.
    os.environ.setdefault("AWS_ACCESS_KEY_ID", KEY_ID)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", SECRET)
    os.environ.setdefault("AWS_REGION", "auto")
    os.environ.setdefault("AWS_ENDPOINT", ENDPOINT_URL)
    os.environ.setdefault("AWS_ENDPOINT_URL", ENDPOINT_URL)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL lance; LOAD lance;")
    # DuckDB secret: used by httpfs for the glob (lance uses the AWS_* env above).
    con.execute(
        "CREATE OR REPLACE SECRET r2 (TYPE s3, PROVIDER config, KEY_ID ?, SECRET ?, "
        "ENDPOINT ?, REGION 'auto', URL_STYLE 'path', USE_SSL true)",
        [KEY_ID, SECRET, HOST],
    )
    return con


def discover_roots(con) -> list[str]:
    union = " UNION ALL ".join(
        f"SELECT file FROM glob('{ACTIVE_PREFIX}/{'*/' * d}_versions/*.manifest')"
        for d in range(1, MAX_DEPTH + 1)
    )
    rows = con.execute(union).fetchall()
    return sorted({f.rsplit("/_versions/", 1)[0] for (f,) in rows})


def generate_catalog() -> dict:
    con = _connect()
    roots = discover_roots(con)
    print(f"discovered {len(roots)} dataset roots under {ACTIVE_PREFIX}", file=sys.stderr)

    domains: dict[str, list[dict]] = {}
    described = skipped = 0
    for root in roots:
        domain, name = _domain_of(root)
        try:
            cols = con.execute(f"DESCRIBE SELECT * FROM __lance_scan('{root}')").fetchall()
            schema = {c[0]: c[1] for c in cols}
            described += 1
        except Exception as exc:  # malformed/unreadable dataset — keep it listed, no schema
            schema = {}
            skipped += 1
            print(f"  skip schema for {name}: {str(exc).splitlines()[0][:120]}", file=sys.stderr)
        domains.setdefault(domain, []).append(
            {"dataset_name": name, "path": root, "schema": schema}
        )

    print(f"described {described}, schema-less {skipped}", file=sys.stderr)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active_prefix": ACTIVE_PREFIX,
        "dataset_count": len(roots),
        "domains": domains,
    }


def main() -> None:
    catalog = generate_catalog()
    body = json.dumps(catalog, indent=2).encode("utf-8")
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET,
        region_name="auto",
    )
    s3.put_object(Bucket=BUCKET, Key=CATALOG_KEY, Body=body, ContentType="application/json")
    print(f"wrote s3://{BUCKET}/{CATALOG_KEY} ({len(body)} bytes, {catalog['dataset_count']} datasets)")


if __name__ == "__main__":
    main()
