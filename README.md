# bi-ui — Metabase + DuckDB Driver

Custom Metabase container image that injects the community-maintained DuckDB JDBC
driver, enabling Metabase to connect to the upstream Quack server.

The image is built from [`metabase/metabase:latest`](https://hub.docker.com/r/metabase/metabase)
and adds [`metabase-core-duckdb-driver`](https://github.com/AlexR2D2/metabase-core-duckdb-driver)
into the plugins directory at build time.

## Build

```bash
docker build -t bi-ui .
```

## Render Deployment

Deploy as a Render Web Service built directly from this repository's `Dockerfile`.

| Setting | Value |
| --- | --- |
| **Runtime** | Docker |
| **Port** | `PORT=3000` |
| **Instance Size** | Starter ($7/mo) or Standard ($25/mo) |

### Runtime

Set the service **Runtime** to **Docker**. Render builds the image from the
`Dockerfile` at the repository root — no build command or start command required.

### Port Mapping

Metabase listens on port `3000`. Expose it to Render by setting the environment
variable:

```
PORT=3000
```

### Instance Size

Metabase runs on the JVM and is memory-bound. Choose an instance with enough
headroom for the JVM heap:

- **Starter — $7/mo:** minimum viable footprint for light usage and evaluation.
- **Standard — $25/mo:** recommended for sustained query load to satisfy the
  JVM's memory requirements and avoid out-of-memory restarts.

Free / 512 MB tiers are insufficient and will cause the container to be OOM-killed
during startup.
