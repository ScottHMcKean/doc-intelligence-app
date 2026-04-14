"""Entry points for Databricks Asset Bundle jobs."""

import argparse
import logging
import sys
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState

from doc_intel.config import load_config
from doc_intel.database.service import DatabaseService
from doc_intel.document.service import DocumentService
from doc_intel.agent.service import AgentService
from doc_intel.integration.service_integration import ServiceIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_workspace_client() -> WorkspaceClient:
    """Get configured Databricks workspace client."""
    try:
        return WorkspaceClient()
    except Exception as e:
        logger.error(f"Failed to initialize workspace client: {e}")
        sys.exit(1)


def database_setup():
    """Entry point for database setup job."""
    parser = argparse.ArgumentParser(description="Setup database instance and tables")
    parser.add_argument("--environment", required=True, help="Environment (dev/prod)")
    args = parser.parse_args()

    logger.info(f"Starting database setup for environment: {args.environment}")

    try:
        # Load configuration
        config = load_config()

        # Initialize services
        client = get_workspace_client()
        db_service = DatabaseService(client, config)

        # Setup database instance
        logger.info("Setting up database instance...")
        if not db_service.setup_database_instance():
            logger.error("Failed to setup database instance")
            sys.exit(1)

        # Create tables
        logger.info("Creating database tables...")
        if not db_service.create_tables():
            logger.error("Failed to create tables")
            sys.exit(1)

        # Create user
        logger.info("Creating user...")
        user = db_service.create_user()
        if not user:
            logger.error("Failed to create user")
            sys.exit(1)

        logger.info(f"Database setup completed successfully for user: {user.username}")

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)


def document_processing():
    """Entry point for document processing job."""
    parser = argparse.ArgumentParser(description="Process document and store chunks")
    parser.add_argument("--environment", required=True, help="Environment (dev/prod)")
    parser.add_argument("--input_path", required=True, help="Input document path")
    parser.add_argument("--output_path", required=True, help="Output processed path")
    parser.add_argument("--doc_hash", required=True, help="Document hash")
    args = parser.parse_args()

    logger.info(f"Starting document processing for: {args.doc_hash}")

    try:
        # Load configuration
        config = load_config()

        # Initialize integrated service
        client = get_workspace_client()
        service_integration = ServiceIntegration(client, config)

        # Process document with database integration
        success, document_id, message = (
            service_integration.process_document_with_database(
                input_path=args.input_path,
                output_path=args.output_path,
                doc_hash=args.doc_hash,
                metadata={"environment": args.environment},
            )
        )

        if not success:
            logger.error(f"Document processing failed: {message}")
            sys.exit(1)

        logger.info(f"Document processing completed successfully: {message}")

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        sys.exit(1)


def agent_workflow():
    """Entry point for agent workflow job."""
    parser = argparse.ArgumentParser(description="Run agent workflow")
    parser.add_argument("--environment", required=True, help="Environment (dev/prod)")
    parser.add_argument("--conversation_id", required=True, help="Conversation ID")
    parser.add_argument(
        "--user_message", default="Hello", help="User message to process"
    )
    args = parser.parse_args()

    logger.info(f"Starting agent workflow for conversation: {args.conversation_id}")

    try:
        # Load configuration
        config = load_config()

        # Initialize integrated service
        client = get_workspace_client()
        service_integration = ServiceIntegration(client, config)

        # Run agent workflow with database integration
        success, response, message = (
            service_integration.run_agent_workflow_with_database(
                conversation_id=args.conversation_id, user_message=args.user_message
            )
        )

        if not success:
            logger.error(f"Agent workflow failed: {message}")
            sys.exit(1)

        logger.info(f"Agent workflow completed successfully: {message}")

    except Exception as e:
        logger.error(f"Agent workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # This allows running the module directly for testing
    if len(sys.argv) > 1:
        if sys.argv[1] == "database_setup":
            database_setup()
        elif sys.argv[1] == "document_processing":
            document_processing()
        elif sys.argv[1] == "agent_workflow":
            agent_workflow()
        else:
            print(
                "Unknown command. Use: database_setup, document_processing, or agent_workflow"
            )
            sys.exit(1)
    else:
        print("Usage: python -m doc_intel.entry_points <command>")
        sys.exit(1)
