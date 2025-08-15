"""Serverless job queueing for document processing."""

import logging
from typing import Optional, Dict, Any

import streamlit as st
from databricks.sdk.service.jobs import NotebookTask, JobSettings, Task

from ..auth import get_databricks_client
from ..config import config

logger = logging.getLogger(__name__)


def queue_document_processing(
    input_path: str,
    output_path: str,
    doc_hash: str,
    job_cluster_key: str = "default_cluster",
    notebook_path: Optional[str] = None,
) -> Optional[int]:
    """
    Queue a serverless job to process the document using ai_parse_document with graceful error handling.

    Args:
        input_path: Path to the uploaded document in volume
        output_path: Path where processed document should be saved
        doc_hash: Unique hash for the document
        job_cluster_key: Cluster configuration key
        notebook_path: Path to the document processing notebook

    Returns:
        Job run ID if successful, None otherwise
    """
    # Use configured job ID or default notebook path
    if config.databricks_job_id:
        # If a pre-configured job exists, use it
        return queue_existing_job(input_path, output_path, doc_hash)
    
    notebook_path = notebook_path or "/Workspace/Users/shared/document_processor"
    
    # Check if Databricks is available
    if not config.databricks_available:
        logger.warning("Databricks not available, simulating job queue")
        st.warning("⚠️ Databricks not connected. Document processing will be simulated.")
        import random
        run_id = random.randint(10000, 99999)
        st.info(f"📝 Simulated processing job queued. Run ID: {run_id}")
        return run_id

    try:
        client = get_databricks_client()
        if not client:
            logger.error("Failed to get Databricks client for job queue")
            st.error("❌ Unable to connect to Databricks for job queue")
            return None

        # Define the notebook task with parameters
        notebook_task = NotebookTask(
            notebook_path=notebook_path,
            base_parameters={
                "input_path": input_path,
                "output_path": output_path,
                "doc_hash": doc_hash,
            },
        )

        # Create job configuration
        job_settings = JobSettings(
            name=f"Document Processing - {doc_hash}",
            tasks=[
                Task(
                    task_key="process_document",
                    notebook_task=notebook_task,
                    job_cluster_key=job_cluster_key,
                )
            ],
            job_clusters=[
                {
                    "job_cluster_key": job_cluster_key,
                    "new_cluster": {
                        "spark_version": "13.3.x-scala2.12",
                        "node_type_id": "i3.xlarge",
                        "num_workers": 0,  # Serverless
                        "spark_conf": {
                            "spark.databricks.cluster.profile": "serverless"
                        },
                    },
                }
            ],
            timeout_seconds=1800,  # 30 minutes timeout
        )

        # Create and run the job
        logger.info(f"Creating job for document processing: {doc_hash}")
        job_response = client.jobs.create(job_settings)
        job_id = job_response.job_id

        # Run the job
        run_response = client.jobs.run_now(
            job_id=job_id,
            notebook_params={
                "input_path": input_path,
                "output_path": output_path,
                "doc_hash": doc_hash,
            },
        )

        run_id = run_response.run_id
        logger.info(f"Successfully queued job {run_id} for document {doc_hash}")
        st.success(f"✅ Document processing job queued. Run ID: {run_id}")

        return run_id

    except Exception as e:
        logger.error(f"Failed to queue document processing job: {str(e)}")
        st.error(f"❌ Failed to queue document processing job: {str(e)}")
        return None


def queue_existing_job(input_path: str, output_path: str, doc_hash: str) -> Optional[int]:
    """Queue processing using an existing Databricks job."""
    try:
        client = get_databricks_client()
        if not client:
            logger.error("Failed to get Databricks client for existing job")
            return None

        job_id = int(config.databricks_job_id)
        logger.info(f"Using existing job {job_id} for document processing")

        # Run the existing job with parameters
        run_response = client.jobs.run_now(
            job_id=job_id,
            notebook_params={
                "input_path": input_path,
                "output_path": output_path,
                "doc_hash": doc_hash,
            },
        )

        run_id = run_response.run_id
        logger.info(f"Successfully queued existing job {run_id} for document {doc_hash}")
        st.success(f"✅ Processing job queued using existing job. Run ID: {run_id}")

        return run_id

    except Exception as e:
        logger.error(f"Failed to queue existing job: {str(e)}")
        st.error(f"❌ Failed to queue existing job: {str(e)}")
        return None


def check_job_status(run_id: int) -> Optional[Dict[str, Any]]:
    """
    Check the status of a document processing job with graceful error handling.

    Returns:
        Job status information if successful, None otherwise
    """
    # Check if Databricks is available
    if not config.databricks_available:
        logger.warning("Databricks not available, returning simulated job status")
        return {
            "run_id": run_id,
            "state": "TERMINATED",
            "result_state": "SUCCESS",
            "start_time": None,
            "end_time": None,
            "run_page_url": None,
            "simulated": True
        }
    
    try:
        client = get_databricks_client()
        if not client:
            logger.error("Failed to get Databricks client for job status check")
            return None
            
        run_info = client.jobs.get_run(run_id)
        
        status_info = {
            "run_id": run_id,
            "state": (
                run_info.state.life_cycle_state.value if run_info.state else "UNKNOWN"
            ),
            "result_state": (
                run_info.state.result_state.value
                if run_info.state and run_info.state.result_state
                else None
            ),
            "start_time": run_info.start_time,
            "end_time": run_info.end_time,
            "run_page_url": run_info.run_page_url,
            "simulated": False
        }
        
        logger.info(f"Job {run_id} status: {status_info['state']}")
        return status_info

    except Exception as e:
        logger.error(f"Failed to check job status for run {run_id}: {str(e)}")
        st.error(f"❌ Failed to check job status: {str(e)}")
        return None


def create_processing_notebook() -> str:
    """
    Return the notebook code for document processing.
    This should be deployed to Databricks as a notebook.
    """
    notebook_code = '''
# Databricks notebook source
# Document Processing Notebook
# This notebook processes uploaded documents using ai_parse_document

# COMMAND ----------

# Get parameters
dbutils.widgets.text("input_path", "")
dbutils.widgets.text("output_path", "") 
dbutils.widgets.text("doc_hash", "")

input_path = dbutils.widgets.get("input_path")
output_path = dbutils.widgets.get("output_path")
doc_hash = dbutils.widgets.get("doc_hash")

print(f"Processing document: {input_path}")
print(f"Output path: {output_path}")
print(f"Document hash: {doc_hash}")

# COMMAND ----------

import json
from pathlib import Path

# Read the document
with open(input_path, 'rb') as f:
    document_content = f.read()

# Use ai_parse_document to process the document
try:
    # Note: This is a placeholder - replace with actual ai_parse_document call
    parsed_result = spark.sql(f"""
        SELECT ai_parse_document('{input_path}') as parsed_content
    """).collect()[0]['parsed_content']
    
    # Prepare output data
    output_data = {
        "doc_hash": doc_hash,
        "input_path": input_path,
        "processed_at": str(spark.sql("SELECT current_timestamp()").collect()[0][0]),
        "parsed_content": parsed_result,
        "status": "success"
    }
    
except Exception as e:
    print(f"Error processing document: {str(e)}")
    output_data = {
        "doc_hash": doc_hash,
        "input_path": input_path,
        "processed_at": str(spark.sql("SELECT current_timestamp()").collect()[0][0]),
        "error": str(e),
        "status": "error"
    }

# COMMAND ----------

# Save processed result
output_file = f"{output_path.rstrip('/')}/{doc_hash}_processed.json"

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Processing complete. Output saved to: {output_file}")
'''
    return notebook_code
