#!/usr/bin/env python3
"""
Test script for notebook deployment functionality.

This script demonstrates how to use the new deploy_notebook_to_workspace method
in the DocumentService to deploy the ai_parse.py script to Databricks workspace.
"""

import sys
import os
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from doc_intel.document.service import DocumentService
from doc_intel.config import DocConfig
from doc_intel.utils import get_workspace_client


def test_notebook_deployment():
    """Test deploying the ai_parse.py notebook to Databricks workspace."""

    # Initialize configuration
    config = DocConfig("./config.yaml")

    # Get workspace client
    client = get_workspace_client()
    if not client:
        print("❌ Failed to get Databricks workspace client")
        return False

    # Initialize document service
    doc_service = DocumentService(client=client, config=config)

    # Paths
    local_notebook_path = "src/doc_intel/document/ai_parse.py"
    workspace_path = "/Workspace/Shared/ai_parse_document_processing"

    # Check if local file exists
    if not os.path.exists(local_notebook_path):
        print(f"❌ Local notebook file not found: {local_notebook_path}")
        return False

    print(f"📁 Local notebook: {local_notebook_path}")
    print(f"🎯 Target workspace path: {workspace_path}")

    # Deploy notebook to workspace
    print("\n🚀 Deploying notebook to Databricks workspace...")
    success, job_id, message = doc_service.deploy_notebook_to_workspace(
        local_notebook_path=local_notebook_path,
        workspace_path=workspace_path,
        create_job=True,  # Also create a job for the notebook
    )

    if success:
        print(f"✅ {message}")
        if job_id:
            print(f"📋 Job created with ID: {job_id}")
        print(f"\n🔗 You can now find your notebook at: {workspace_path}")
        print("💡 The notebook is ready to be used in Databricks jobs!")
        return True
    else:
        print(f"❌ {message}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Notebook Deployment Functionality")
    print("=" * 50)

    success = test_notebook_deployment()

    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)
