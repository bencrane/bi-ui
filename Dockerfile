FROM metabase/metabase:latest

# Download the community DuckDB driver into the Metabase plugins directory
ADD https://github.com/AlexR2D2/metabase-core-duckdb-driver/releases/latest/download/duckdb.metabase-driver.jar /plugins/

# Ensure proper permissions for the JVM to load the plugin on startup
RUN chmod 777 /plugins/duckdb.metabase-driver.jar
