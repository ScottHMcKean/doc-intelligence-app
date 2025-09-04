# Databricks notebook source
# MAGIC %md
# MAGIC # Document Service
# MAGIC
# MAGIC This notebook forms the core of our document service. It showcases how we are going to simplify our document intelligence application using Lakebase and Serverless jobs. This is tested on Serverless Version 3 - it takes a single file or a directory and parses all the files directly into an append operation on a postgres table. We can then get embeddings and use pgvector as the backend with a langgraph Agent.

# COMMAND ----------

# MAGIC %md
# MAGIC We use our Databricks user IDs as the main entry point into the workflow and authentication

# COMMAND ----------

# MAGIC %pip install databricks-langchain databricks-sdk --upgrade
# MAGIC %restart_python

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
me = w.current_user.me()
print(me.id)  # This is your Databricks user ID
print(me.user_name)
USER_ID = me.id

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql.types import *

dbutils.widgets.text("file_path", "/Volumes/main/default/raw_pdfs/test.pdf")
dbutils.widgets.text("embedding_endpoint", "databricks-gte-large-en")
dbutils.widgets.text("database_instance", "shm")

# COMMAND ----------

config = dbutils.widgets.getAll()
config

# COMMAND ----------

# MAGIC %md
# MAGIC We use ai_parse_document in a serverless job as our document processing service. This could be any isolated microservice and has lots of room for optimization, but ai_parse_document does a pretty good job and can handle lots of file types

# COMMAND ----------

parsed_df = (
    spark.read.format("binaryFile")
    .load(config.get("file_path"))
    .withColumn("user_id", lit(USER_ID))
    .select(
        col("path"), col("user_id"), expr("ai_parse_document(content)").alias("parsed")
    )
    .withColumn("parsed_json", parse_json(col("parsed").cast("string")))
    .select(
        col("path"),
        col("user_id"),
        expr("parsed_json:document:pages").alias("pages"),
        expr("parsed_json:document:elements").alias("elements"),
        expr("parsed_json:document:_corrupted_data").alias("_corrupted_data"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC To get something simple and working, I propose that we simply chunk each page for now. We can work on refining the chunking strategy in this job, but this gives a good starting point. We even wrap the embedding call here for better horizontal scalability.

# COMMAND ----------

from pyspark.sql.functions import from_json, explode, col, concat_ws, lit
from pyspark.sql.types import (
    ArrayType,
    StructType,
    StructField,
    IntegerType,
    StringType,
)

# Define schema for pages based on provided example
page_schema = StructType(
    [
        StructField("content", StringType()),
        StructField("footer", StringType()),
        StructField("header", StringType()),
        StructField("id", IntegerType()),
        StructField("page_number", IntegerType()),
    ]
)

chunked_pages = (
    parsed_df.withColumn(
        "pages_array", from_json(col("pages").cast("string"), ArrayType(page_schema))
    )
    .withColumn("page_chunk", explode(col("pages_array")))
    .select(
        col("path"),
        col("user_id"),
        col("page_chunk.id").cast("string").alias("page_id"),
        concat_ws(
            "\n",
            concat_ws("", lit("Content: ["), col("page_chunk.content"), lit("]")),
            concat_ws("", lit("Footer: ["), col("page_chunk.footer"), lit("]")),
            concat_ws("", lit("Header: ["), col("page_chunk.header"), lit("]")),
            concat_ws("", lit("ID: ["), col("page_chunk.id").cast("string"), lit("]")),
            concat_ws(
                "",
                lit("Page Number: ["),
                col("page_chunk.page_number").cast("string"),
                lit("]"),
            ),
        ).alias("text"),
    )
    .withColumn(
        "embedding", expr(f"ai_query('{config.get('embedding_endpoint')}', text)")
    )
)

# COMMAND ----------

chunked_pages_pd = chunked_pages.toPandas()
chunked_pages_pd["embedding"] = chunked_pages_pd["embedding"].apply(lambda x: list(x))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Postgres Connection
# MAGIC Now we are going to move our chunks into postgres. First we create our chunks table, next we read into Pandas and write chunks into Postgres. We can bolster the connection and database for horizontal scalability (e.g. https://learn.microsoft.com/en-us/azure/databricks/oltp/query/notebook)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.database import DatabaseInstance

import psycopg2
import uuid

w = WorkspaceClient()

db_instance_name = config.get("database_instance")
try:
    instance = w.database.get_database_instance(name=db_instance_name)
    print(f"Existing database instance found: {instance.read_write_dns}")
except Exception:
    print(f"Database instance not found! {instance.name}")
    w.database.create_database_instance(
        DatabaseInstance(name=db_instance_name, capacity="CU_2")
    )
    instance = w.database.get_database_instance(name=db_instance_name)

    CRED = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()), instance_names=[db_instance_name]
    )


def connect_to_pg():
    conn = psycopg2.connect(
        host=instance.read_write_dns,
        dbname="databricks_postgres",
        user=me.user_name,
        password=CRED.token,
        sslmode="require",
    )
    return conn


def run_pg_query(query, data_tuples=None):
    conn = connect_to_pg()
    with conn.cursor() as cur:
        if data_tuples:
            cur.executemany(query, data_tuples)
        else:
            cur.execute(query)
        conn.commit()
    conn.close()
    return True


# COMMAND ----------

# MAGIC %md
# MAGIC Make our table if it doesn't exist. This takes <10 μs so isn't a huge production risk to run with every job. Note the unique constraint to avoid duplication of data.

# COMMAND ----------

# MAGIC %time
# MAGIC run_pg_query("CREATE EXTENSION IF NOT EXISTS vector;")
# MAGIC
# MAGIC run_pg_query(
# MAGIC     """
# MAGIC     CREATE TABLE IF NOT EXISTS parsed_pages (
# MAGIC         path TEXT,
# MAGIC         user_id TEXT,
# MAGIC         page_id TEXT,
# MAGIC         text TEXT,
# MAGIC         embedding VECTOR(1024),
# MAGIC         CONSTRAINT user_path_page_pk PRIMARY KEY (user_id, path, page_id)
# MAGIC     );
# MAGIC     """
# MAGIC     )

# COMMAND ----------

# MAGIC %md
# MAGIC Insert records into our PG table. We keep appending the table (could have individual user tables but this is unnecessary in my opinion).

# COMMAND ----------

data_tuples = [tuple(x) for x in chunked_pages_pd.to_numpy()]
insert_query = """
  INSERT INTO parsed_pages (path, user_id, page_id, text, embedding) 
  VALUES (%s, %s, %s, %s, %s)
  ON CONFLICT (path, user_id, page_id) DO NOTHING
  """
run_pg_query(insert_query, data_tuples)
