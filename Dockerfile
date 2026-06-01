FROM metabase/metabase:latest

# Explicitly define where Metabase should look for plugins
ENV MB_PLUGINS_DIR=/plugins/

# Briefly switch to root to set up the directory structure and permissions
USER root
RUN mkdir -p /plugins/

# Download the actively maintained MotherDuck driver
ADD https://github.com/MotherDuck-Open-Source/metabase_duckdb_driver/releases/latest/download/duckdb.metabase-driver.jar /plugins/

# Grant the metabase user (UID 2000) full ownership and execution rights
RUN chmod 777 /plugins/duckdb.metabase-driver.jar && \
    chown -R 2000:2000 /plugins/

# Drop back to the secure, non-root metabase user
USER 2000
