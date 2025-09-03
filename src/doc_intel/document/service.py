"""Document service for Databricks serverless jobs."""

import logging
from typing import Optional, Dict, Any, Tuple
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import NotebookTask, JobSettings, Task
from databricks.sdk.service.workspace import ImportFormat, Language

from doc_intel.config import check_config
from doc_intel.utils import check_workspace_client

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document processing jobs with graceful error handling."""

    def __init__(self, client: Optional[WorkspaceClient], config: dict):
        self.client = client
        check_workspace_client(client)
        self.user_id = self.client.current_user.me().id

        self.config = config
        check_config(config)

        self.timeout_minutes = config.document.get("timeout_minutes", 30)
        self.max_retries = config.document.get("max_retries", 3)
        self.auto_process = config.document.get("auto_process", True)
        self.generate_summary = config.document.get("generate_summary", True)

    def queue_document_processing(
        self,
        input_path: str,
        output_path: str,
        doc_hash: str,
        job_cluster_key: str = "default_cluster",
        notebook_path: Optional[str] = None,
    ) -> Tuple[bool, Optional[int], str]:
        """
        Queue a serverless job to process the document.

        Returns:
            Tuple of (success, run_id, message)
        """
        # Use configured job ID if available
        agent_job_id = self.config.document.get("job_id")
        if agent_job_id:
            return self._queue_existing_job(input_path, output_path, doc_hash)

        notebook_path = notebook_path or self.config.document.get(
            "processing_notebook_path", "/Workspace/notebooks/document_processing"
        )
        job_cluster_key = job_cluster_key or self.config.get(
            "default_cluster_key", "default_cluster"
        )

        try:
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
                timeout_seconds=self.timeout_minutes * 60,
            )

            # Create and run the job
            logger.info(f"Creating job for document processing: {doc_hash}")
            job_response = self.client.jobs.create(job_settings)
            job_id = job_response.job_id

            # Run the job
            run_response = self.client.jobs.run_now(
                job_id=job_id,
                notebook_params={
                    "input_path": input_path,
                    "output_path": output_path,
                    "doc_hash": doc_hash,
                },
            )

            run_id = run_response.run_id
            logger.info(f"Successfully queued job {run_id} for document {doc_hash}")
            return True, run_id, f"Document processing job queued. Run ID: {run_id}"

        except Exception as e:
            logger.error(f"Failed to queue document processing job: {str(e)}")
            return False, None, f"Failed to queue job: {str(e)}"

    def _queue_existing_job(
        self, input_path: str, output_path: str, doc_hash: str
    ) -> Tuple[bool, Optional[int], str]:
        """Queue processing using an existing Databricks job."""
        try:
            job_id = int(self.config.get("job_id"))
            logger.info(f"Using existing job {job_id} for document processing")

            # Run the existing job with parameters
            run_response = self.client.jobs.run_now(
                job_id=job_id,
                notebook_params={
                    "input_path": input_path,
                    "output_path": output_path,
                    "doc_hash": doc_hash,
                },
            )

            run_id = run_response.run_id
            logger.info(
                f"Successfully queued existing job {run_id} for document {doc_hash}"
            )
            return (
                True,
                run_id,
                f"Processing job queued using existing job. Run ID: {run_id}",
            )

        except Exception as e:
            logger.error(f"Failed to queue existing job: {str(e)}")
            return False, None, f"Failed to queue existing job: {str(e)}"

    def check_job_status(
        self, run_id: int
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Check the status of a document processing job.

        Returns:
            Tuple of (success, status_info, message)
        """
        try:
            run_info = self.client.jobs.get_run(run_id)

            status_info = {
                "run_id": run_id,
                "state": (
                    run_info.state.life_cycle_state.value
                    if run_info.state
                    else "UNKNOWN"
                ),
                "result_state": (
                    run_info.state.result_state.value
                    if run_info.state and run_info.state.result_state
                    else None
                ),
                "start_time": run_info.start_time,
                "end_time": run_info.end_time,
                "run_page_url": run_info.run_page_url,
                "simulated": False,
            }

            logger.info(f"Job {run_id} status: {status_info['state']}")
            return True, status_info, "Job status retrieved successfully"

        except Exception as e:
            logger.error(f"Failed to check job status for run {run_id}: {str(e)}")
            return False, None, f"Failed to check job status: {str(e)}"

    def cancel_job(self, run_id: int) -> Tuple[bool, str]:
        """
        Cancel a running job.

        Returns:
            Tuple of (success, message)
        """
        try:
            self.client.jobs.cancel_run(run_id)
            logger.info(f"Successfully cancelled job {run_id}")
            return True, f"Job {run_id} cancelled successfully"
        except Exception as e:
            logger.error(f"Failed to cancel job {run_id}: {str(e)}")
            return False, f"Failed to cancel job: {str(e)}"

    def deploy_notebook_to_workspace(
        self, local_notebook_path: str, workspace_path: str, create_job: bool = False
    ) -> Tuple[bool, Optional[int], str]:
        """
        Deploy a notebook to Databricks workspace.

        Args:
            local_notebook_path: Path to the local .py file
            workspace_path: Target path in Databricks workspace (e.g., "/Workspace/Shared/my_notebook")
            create_job: Whether to create a job for the notebook

        Returns:
            Tuple of (success, job_id, message)
        """
        try:
            # Read the notebook content
            with open(local_notebook_path, "r") as f:
                content = f.read()

            # Upload the notebook to workspace
            self.client.workspace.import_(
                path=workspace_path,
                content=content,
                format=ImportFormat.SOURCE,
                language=Language.PYTHON,
                overwrite=True,
            )

            logger.info(f"Successfully deployed notebook to {workspace_path}")

            job_id = None
            if create_job:
                # Create a job for the notebook
                job_settings = JobSettings(
                    name=f"Notebook Job - {workspace_path.split('/')[-1]}",
                    tasks=[
                        Task(
                            task_key="run_notebook",
                            notebook_task=NotebookTask(
                                notebook_path=workspace_path, base_parameters={}
                            ),
                        )
                    ],
                    timeout_seconds=3600,  # 1 hour timeout
                )

                job_response = self.client.jobs.create(job_settings)
                job_id = job_response.job_id
                logger.info(f"Created job {job_id} for notebook {workspace_path}")

            return True, job_id, f"Notebook deployed to {workspace_path}"

        except Exception as e:
            logger.error(f"Failed to deploy notebook: {str(e)}")
            return False, None, f"Failed to deploy notebook: {str(e)}"
