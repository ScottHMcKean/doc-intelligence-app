"""Integration tests for the DocumentService."""

import pytest
from unittest.mock import Mock, patch
from databricks.sdk.service.jobs import JobSettings, Task, NotebookTask

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from doc_intel.document.service import DocumentService


class TestDocumentService:
    """Integration tests for the document service."""

    def test_document_service_initialization(self, test_config, mock_databricks_client):
        """Test document service initialization."""
        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        assert doc_service.client is not None
        assert doc_service.config is not None
        assert doc_service.timeout_minutes == 5  # From test config
        assert doc_service.max_retries == 2  # From test config
        assert doc_service.auto_process is True

    def test_document_service_initialization_without_client(self, test_config):
        """Test document service initialization without client."""
        doc_service = DocumentService(client=None, config=test_config)

        assert doc_service.client is None
        assert doc_service.config is not None

    def test_queue_document_processing_with_existing_job(
        self, test_config, mock_databricks_client
    ):
        """Test queuing document processing with existing job ID."""
        # Add job_id to config
        test_config["job_id"] = "123"
        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test queuing with existing job
        success, run_id, message = doc_service.queue_document_processing(
            input_path="/path/to/input",
            output_path="/path/to/output",
            doc_hash="test_hash",
        )

        assert success is True
        assert run_id == 456  # From mock
        assert "existing job" in message.lower()

    def test_queue_document_processing_without_client(self, test_config):
        """Test queuing document processing without Databricks client."""
        doc_service = DocumentService(client=None, config=test_config)

        # Test queuing without client
        success, run_id, message = doc_service.queue_document_processing(
            input_path="/path/to/input",
            output_path="/path/to/output",
            doc_hash="test_hash",
        )

        assert success is True
        assert run_id is not None
        assert "simulated" in message.lower()

    def test_queue_document_processing_with_new_job(
        self, test_config, mock_databricks_client
    ):
        """Test queuing document processing with new job creation."""
        # Ensure no job_id in config
        if "job_id" in test_config:
            del test_config["job_id"]

        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test queuing with new job creation
        success, run_id, message = doc_service.queue_document_processing(
            input_path="/path/to/input",
            output_path="/path/to/output",
            doc_hash="test_hash",
        )

        assert success is True
        assert run_id == 456  # From mock
        assert "job queued" in message.lower()

    def test_queue_existing_job(self, test_config, mock_databricks_client):
        """Test queuing using existing job."""
        test_config["job_id"] = "123"
        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test queuing existing job
        success, run_id, message = doc_service._queue_existing_job(
            input_path="/path/to/input",
            output_path="/path/to/output",
            doc_hash="test_hash",
        )

        assert success is True
        assert run_id == 456  # From mock
        assert "existing job" in message.lower()

    def test_queue_existing_job_without_client(self, test_config):
        """Test queuing existing job without client."""
        test_config["job_id"] = "123"
        doc_service = DocumentService(client=None, config=test_config)

        # Test queuing existing job without client
        success, run_id, message = doc_service._queue_existing_job(
            input_path="/path/to/input",
            output_path="/path/to/output",
            doc_hash="test_hash",
        )

        assert success is False
        assert run_id is None
        assert "not available" in message.lower()

    def test_check_job_status(self, test_config, mock_databricks_client):
        """Test checking job status."""
        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test checking job status
        success, status_info, message = doc_service.check_job_status(run_id=456)

        assert success is True
        assert isinstance(status_info, dict)
        assert status_info["run_id"] == 456
        assert status_info["state"] == "TERMINATED"
        assert status_info["result_state"] == "SUCCESS"
        assert status_info["simulated"] is False
        assert "successfully" in message.lower()

    def test_check_job_status_without_client(self, test_config):
        """Test checking job status without client."""
        doc_service = DocumentService(client=None, config=test_config)

        # Test checking job status without client
        success, status_info, message = doc_service.check_job_status(run_id=456)

        assert success is True
        assert isinstance(status_info, dict)
        assert status_info["run_id"] == 456
        assert status_info["state"] == "TERMINATED"
        assert status_info["result_state"] == "SUCCESS"
        assert status_info["simulated"] is True
        assert "simulated" in message.lower()

    def test_cancel_job(self, test_config, mock_databricks_client):
        """Test canceling a job."""
        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test canceling job
        success, message = doc_service.cancel_job(run_id=456)

        assert success is True
        assert "cancelled successfully" in message.lower()

    def test_cancel_job_without_client(self, test_config):
        """Test canceling job without client."""
        doc_service = DocumentService(client=None, config=test_config)

        # Test canceling job without client
        success, message = doc_service.cancel_job(run_id=456)

        assert success is False
        assert "not available" in message.lower()

    def test_cancel_job_failure(self, test_config, mock_databricks_client):
        """Test canceling job with failure."""
        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Mock client to raise exception
        mock_databricks_client.jobs.cancel_run.side_effect = Exception("Cancel failed")

        # Test canceling job with failure
        success, message = doc_service.cancel_job(run_id=456)

        assert success is False
        assert "failed" in message.lower()

    def test_job_creation_with_notebook_task(self, test_config, mock_databricks_client):
        """Test job creation with notebook task."""
        # Ensure no job_id in config
        if "job_id" in test_config:
            del test_config["job_id"]

        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test job creation
        success, run_id, message = doc_service.queue_document_processing(
            input_path="/path/to/input",
            output_path="/path/to/output",
            doc_hash="test_hash",
            notebook_path="/Workspace/test/notebook",
        )

        assert success is True
        assert run_id == 456
        assert "job queued" in message.lower()

    def test_job_creation_with_custom_cluster_key(
        self, test_config, mock_databricks_client
    ):
        """Test job creation with custom cluster key."""
        # Ensure no job_id in config
        if "job_id" in test_config:
            del test_config["job_id"]

        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test job creation with custom cluster key
        success, run_id, message = doc_service.queue_document_processing(
            input_path="/path/to/input",
            output_path="/path/to/output",
            doc_hash="test_hash",
            job_cluster_key="custom_cluster",
        )

        assert success is True
        assert run_id == 456

    def test_job_creation_failure(self, test_config, mock_databricks_client):
        """Test job creation failure."""
        # Ensure no job_id in config
        if "job_id" in test_config:
            del test_config["job_id"]

        # Mock client to raise exception
        mock_databricks_client.jobs.create.side_effect = Exception(
            "Job creation failed"
        )

        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test job creation failure
        success, run_id, message = doc_service.queue_document_processing(
            input_path="/path/to/input",
            output_path="/path/to/output",
            doc_hash="test_hash",
        )

        assert success is False
        assert run_id is None
        assert "failed" in message.lower()

    def test_job_run_failure(self, test_config, mock_databricks_client):
        """Test job run failure."""
        # Ensure no job_id in config
        if "job_id" in test_config:
            del test_config["job_id"]

        # Mock client to raise exception on run_now
        mock_databricks_client.jobs.run_now.side_effect = Exception("Job run failed")

        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test job run failure
        success, run_id, message = doc_service.queue_document_processing(
            input_path="/path/to/input",
            output_path="/path/to/output",
            doc_hash="test_hash",
        )

        assert success is False
        assert run_id is None
        assert "failed" in message.lower()

    def test_existing_job_run_failure(self, test_config, mock_databricks_client):
        """Test existing job run failure."""
        test_config["job_id"] = "123"

        # Mock client to raise exception on run_now
        mock_databricks_client.jobs.run_now.side_effect = Exception("Job run failed")

        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test existing job run failure
        success, run_id, message = doc_service._queue_existing_job(
            input_path="/path/to/input",
            output_path="/path/to/output",
            doc_hash="test_hash",
        )

        assert success is False
        assert run_id is None
        assert "failed" in message.lower()

    def test_job_status_check_failure(self, test_config, mock_databricks_client):
        """Test job status check failure."""
        # Mock client to raise exception
        mock_databricks_client.jobs.get_run.side_effect = Exception(
            "Status check failed"
        )

        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test job status check failure
        success, status_info, message = doc_service.check_job_status(run_id=456)

        assert success is False
        assert status_info is None
        assert "failed" in message.lower()

    def test_configuration_handling(self, test_config):
        """Test configuration handling."""
        doc_service = DocumentService(client=None, config=test_config)

        # Test configuration values
        assert doc_service.timeout_minutes == 5
        assert doc_service.max_retries == 2
        assert doc_service.auto_process is True
        assert doc_service.generate_summary is True

    def test_default_configuration_values(self):
        """Test default configuration values."""
        # Create service with minimal config
        minimal_config = {"timeout_minutes": 10}
        doc_service = DocumentService(client=None, config=minimal_config)

        # Test default values
        assert doc_service.timeout_minutes == 10
        assert doc_service.max_retries == 3  # Default from class
        assert doc_service.auto_process is True  # Default from class
        assert doc_service.generate_summary is True  # Default from class

    def test_notebook_path_configuration(self, test_config, mock_databricks_client):
        """Test notebook path configuration."""
        # Ensure no job_id in config
        if "job_id" in test_config:
            del test_config["job_id"]

        # Set custom notebook path
        test_config["processing_notebook_path"] = "/Workspace/custom/notebook"

        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test with custom notebook path
        success, run_id, message = doc_service.queue_document_processing(
            input_path="/path/to/input",
            output_path="/path/to/output",
            doc_hash="test_hash",
        )

        assert success is True
        assert run_id == 456

    def test_cluster_key_configuration(self, test_config, mock_databricks_client):
        """Test cluster key configuration."""
        # Ensure no job_id in config
        if "job_id" in test_config:
            del test_config["job_id"]

        # Set custom cluster key
        test_config["default_cluster_key"] = "custom_cluster"

        doc_service = DocumentService(client=mock_databricks_client, config=test_config)

        # Test with custom cluster key
        success, run_id, message = doc_service.queue_document_processing(
            input_path="/path/to/input",
            output_path="/path/to/output",
            doc_hash="test_hash",
        )

        assert success is True
        assert run_id == 456
