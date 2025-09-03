#!/usr/bin/env python3
"""
Document Processing Notebook - Converted to Python Script

This script forms the core of our document service. It showcases how we are going to
simplify our document intelligence application using Lakebase and Serverless jobs.
This is tested on Serverless Version 3 - it takes a single file or a directory and
parses all the files directly into an append operation on a postgres table. We can
then get embeddings and use pgvector as the backend with a langgraph Agent.

We use our Databricks user IDs as the main entry point into the workflow and authentication.
"""

# Install required packages
# %pip install databricks-langchain
# %restart_python

import os
import uuid
from typing import Optional, List, Tuple, Any
import psycopg2
import pandas as pd

from databricks.sdk import WorkspaceClient
from databricks_langchain import DatabricksEmbeddings
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


def get_databricks_user_info() -> Tuple[int, str]:
    """Get Databricks user ID and username."""
    w = WorkspaceClient()
    me = w.current_user.me()
    print(f"User ID: {me.id}")
    print(f"Username: {me.user_name}")
    return me.id, me.user_name


def parse_document_with_ai(
    spark: SparkSession,
    volume_path: str,
    user_id: int,
    embedding_endpoint: str = "databricks-gte-large-en",
) -> pd.DataFrame:
    """
    Parse document using AI and create embeddings.

    Args:
        spark: Spark session
        volume_path: Path to the document in volume
        user_id: Databricks user ID
        embedding_endpoint: Embedding model endpoint

    Returns:
        DataFrame with parsed and embedded content
    """
    # Parse document using ai_parse_document
    parsed_df = (
        spark.read.format("binaryFile")
        .load(volume_path)
        .withColumn("user_id", lit(user_id))
        .select(
            col("path"),
            col("user_id"),
            expr("ai_parse_document(content)").alias("parsed"),
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

    # Define schema for pages
    page_schema = StructType(
        [
            StructField("content", StringType()),
            StructField("footer", StringType()),
            StructField("header", StringType()),
            StructField("id", IntegerType()),
            StructField("page_number", IntegerType()),
        ]
    )

    # Chunk pages and generate embeddings
    chunked_pages = (
        parsed_df.withColumn(
            "pages_array",
            from_json(col("pages").cast("string"), ArrayType(page_schema)),
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
                concat_ws(
                    "", lit("ID: ["), col("page_chunk.id").cast("string"), lit("]")
                ),
                concat_ws(
                    "",
                    lit("Page Number: ["),
                    col("page_chunk.page_number").cast("string"),
                    lit("]"),
                ),
            ).alias("text"),
        )
        .withColumn("embedding", expr(f"ai_query('{embedding_endpoint}', 'text')"))
    )

    # Convert to Pandas for easier handling
    chunked_pages_pd = chunked_pages.toPandas()
    chunked_pages_pd["embedding"] = chunked_pages_pd["embedding"].apply(
        lambda x: list(x)
    )

    return chunked_pages_pd


def setup_postgres_connection(
    instance_name: str, user_name: str
) -> psycopg2.extensions.connection:
    """
    Set up PostgreSQL connection using Databricks Lakebase.

    Args:
        instance_name: Name of the database instance
        user_name: Databricks username

    Returns:
        PostgreSQL connection object
    """
    w = WorkspaceClient()

    instance = w.database.get_database_instance(name=instance_name)
    cred = w.database.generate_database_credential(
        request_id=str(uuid.uuid4()), instance_names=[instance_name]
    )

    conn = psycopg2.connect(
        host=instance.read_write_dns,
        dbname="databricks_postgres",
        user=user_name,
        password=cred.token,
        sslmode="require",
    )
    return conn


def create_parsed_pages_table(conn: psycopg2.extensions.connection) -> None:
    """Create the parsed_pages table if it doesn't exist."""
    table_ddl = """
    CREATE TABLE IF NOT EXISTS parsed_pages (
        path TEXT,
        user_id TEXT,
        page_id TEXT,
        text TEXT,
        embedding VECTOR(1024)
    );
    """

    with conn.cursor() as cur:
        cur.execute(table_ddl)
        conn.commit()


def insert_parsed_data(
    conn: psycopg2.extensions.connection, data: pd.DataFrame
) -> None:
    """Insert parsed data into PostgreSQL table."""
    with conn.cursor() as cur:
        records = data.to_records(index=False)
        data_tuples = list(records)
        insert_query = "INSERT INTO parsed_pages (path, user_id, page_id, text, embedding) VALUES (%s, %s, %s, %s, %s)"
        cur.executemany(insert_query, data_tuples)
        conn.commit()


def test_vector_search(
    conn: psycopg2.extensions.connection, query_text: str = "hello world"
) -> List[Tuple]:
    """
    Test vector search functionality.

    Args:
        conn: PostgreSQL connection
        query_text: Text to search for

    Returns:
        List of search results
    """
    # Generate embedding for query
    emb = DatabricksEmbeddings(endpoint="databricks-gte-large-en")
    vect = emb.embed_documents([query_text])[0]

    with conn.cursor() as cur:
        search_query = f"""
            SELECT path, user_id, page_id, text, embedding,
                    (embedding <=> ARRAY{str(vect)}::vector) AS distance
            FROM parsed_pages
            ORDER BY distance ASC
            LIMIT 1
        """
        cur.execute(search_query)
        results = cur.fetchall()

    return results


def main(
    volume_path: str,
    instance_name: str = "shm",
    embedding_endpoint: str = "databricks-gte-large-en",
) -> None:
    """
    Main function to process a document through the AI parsing pipeline.

    Args:
        volume_path: Path to the document in volume
        instance_name: Name of the database instance
        embedding_endpoint: Embedding model endpoint
    """
    # Get user information
    user_id, user_name = get_databricks_user_info()

    # Setup Spark session
    spark = setup_spark_session()

    try:
        # Parse document with AI
        print("Parsing document with AI...")
        parsed_data = parse_document_with_ai(
            spark, volume_path, user_id, embedding_endpoint
        )
        print(f"Parsed {len(parsed_data)} chunks")

        # Setup PostgreSQL connection
        print("Setting up PostgreSQL connection...")
        conn = setup_postgres_connection(instance_name, user_name)

        # Create table if needed
        print("Creating table if needed...")
        create_parsed_pages_table(conn)

        # Insert data
        print("Inserting parsed data...")
        insert_parsed_data(conn, parsed_data)

        # Test vector search
        print("Testing vector search...")
        results = test_vector_search(conn)
        print(f"Search results: {results}")

        print("Document processing completed successfully!")

    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise
    finally:
        # Cleanup
        if "conn" in locals():
            conn.close()
        spark.stop()


if __name__ == "__main__":
    # Example usage - these would typically be passed as parameters
    # when running as a Databricks job
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python ai_parse.py <volume_path> [instance_name] [embedding_endpoint]"
        )
        sys.exit(1)

    volume_path = sys.argv[1]
    instance_name = sys.argv[2] if len(sys.argv) > 2 else "shm"
    embedding_endpoint = sys.argv[3] if len(sys.argv) > 3 else "databricks-gte-large-en"

    main(volume_path, instance_name, embedding_endpoint)
