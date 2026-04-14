# Databricks Asset Bundle (DAB) Setup

This document explains how to use Databricks Asset Bundle (DAB) to manage job creation, serverless environment specifications, and service integration in the Document Intelligence application.

## Overview

The DAB setup provides:
- **Job Management**: Automated creation and deployment of Databricks jobs
- **Serverless Environments**: Optimized serverless compute configurations
- **Service Integration**: Seamless integration between Database and Document services
- **Environment Management**: Separate dev and prod configurations

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │    │   Integration    │    │   Database      │
│   Service       │◄──►│   Layer          │◄──►│   Service       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌──────────────────┐
                    │   Agent Service  │
                    └──────────────────┘
```

## Configuration Files

### `databricks.yml`
Main DAB configuration file defining:
- Bundle metadata and Git integration
- Target environments (dev/prod)
- Job definitions with serverless clusters
- Environment-specific variables

### `resources/`
- `environment.yml`: Environment-specific variables
- `jobs.yml`: Reusable job configurations

## Jobs

### 1. Database Setup (`database_setup`)
- **Purpose**: Initialize database instance and create tables
- **Entry Point**: `doc_intel.entry_points:database_setup`
- **Environment**: Both dev and prod
- **Timeout**: 30 minutes

### 2. Document Processing (`document_processing`)
- **Purpose**: Process documents and store chunks in PostgreSQL
- **Entry Point**: `doc_intel.entry_points:document_processing`
- **Parameters**: `input_path`, `output_path`, `doc_hash`
- **Integration**: Uses Database Service to store chunks

### 3. Agent Workflow (`agent_workflow`)
- **Purpose**: Run conversational AI workflows
- **Entry Point**: `doc_intel.entry_points:agent_workflow`
- **Parameters**: `conversation_id`, `user_message`
- **Integration**: Uses Database Service for conversation storage

## Service Integration

The `ServiceIntegration` class provides a unified interface for:
- **Document Processing**: Process documents and store chunks in PostgreSQL
- **Agent Workflows**: Manage conversations and messages
- **Database Operations**: Unified access to all database operations

### Key Features:
- **Automatic User Management**: Creates users as needed
- **Document Chunk Storage**: Integrates document processing with database storage
- **Conversation Management**: Handles conversation and message persistence
- **Error Handling**: Comprehensive error handling and logging

## Usage

### Prerequisites

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Configure Databricks CLI**:
   ```bash
   databricks configure --token
   ```

3. **Update Configuration**:
   - Update `databricks.yml` with your workspace URL
   - Update Git URL in bundle configuration

### Deployment

1. **Validate Bundle**:
   ```bash
   python scripts/deploy.py --action validate
   ```

2. **Build and Deploy**:
   ```bash
   python scripts/deploy.py --action deploy --target dev
   ```

3. **Run Jobs**:
   ```bash
   # Setup database
   python scripts/deploy.py --action run-job --job-name database_setup --target dev
   
   # Process document
   python scripts/deploy.py --action run-job --job-name document_processing --target dev \
     --parameters input_path=/path/to/doc.pdf output_path=/path/to/processed doc_hash=abc123
   
   # Run agent workflow
   python scripts/deploy.py --action run-job --job-name agent_workflow --target dev \
     --parameters conversation_id=conv123 user_message="Hello"
   ```

### Direct CLI Usage

```bash
# Validate bundle
databricks bundle validate

# Deploy to dev
databricks bundle deploy --target dev

# Deploy to prod
databricks bundle deploy --target prod

# Run specific job
databricks bundle run database_setup --target dev
```

## Environment Variables

### Development
- `environment`: "dev"
- `database_instance_name`: "doc-intel-dev"
- `database_capacity`: "2"
- `max_concurrent_runs`: 3

### Production
- `environment`: "prod"
- `database_instance_name`: "doc-intel-prod"
- `database_capacity`: "4"
- `max_concurrent_runs`: 10

## Serverless Configuration

All jobs use optimized serverless clusters:
- **Spark Version**: 13.3.x-scala2.12
- **Node Type**: i3.xlarge
- **Workers**: 0 (serverless)
- **Data Security**: SINGLE_USER
- **Runtime Engine**: PHOTON

## Integration Examples

### Document Processing with Database Storage

```python
from doc_intel.integration import ServiceIntegration

# Initialize integrated service
service = ServiceIntegration()

# Process document and store chunks
success, document_id, message = service.process_document_with_database(
    input_path="/path/to/document.pdf",
    output_path="/path/to/processed",
    doc_hash="unique_hash",
    metadata={"source": "upload"}
)
```

### Agent Workflow with Database

```python
# Run agent workflow with conversation storage
success, response, message = service.run_agent_workflow_with_database(
    conversation_id="conv_123",
    user_message="What documents do I have?",
    doc_ids=["doc_1", "doc_2"]
)
```

## Monitoring and Logging

- **Job Logs**: Available in Databricks UI under Jobs
- **Application Logs**: Structured logging with timestamps
- **Error Handling**: Comprehensive error messages and stack traces

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   - Ensure Databricks CLI is configured correctly
   - Check workspace URL in `databricks.yml`

2. **Job Failures**:
   - Check job logs in Databricks UI
   - Verify database instance exists and is accessible
   - Ensure all dependencies are installed

3. **Database Connection Issues**:
   - Verify database instance is running
   - Check connection parameters in configuration
   - Ensure PostgreSQL extensions are installed

### Debug Mode

Enable debug logging by setting environment variable:
```bash
export LOG_LEVEL=DEBUG
```

## Best Practices

1. **Environment Separation**: Always use appropriate target (dev/prod)
2. **Resource Management**: Monitor concurrent runs and timeouts
3. **Error Handling**: Implement proper error handling in custom logic
4. **Testing**: Test jobs in dev environment before deploying to prod
5. **Monitoring**: Set up alerts for job failures and performance issues

## Next Steps

1. **Customize Document Processing**: Implement your specific document processing logic
2. **Enhance Agent Workflows**: Add your conversational AI logic
3. **Add Monitoring**: Set up comprehensive monitoring and alerting
4. **Scale Configuration**: Adjust cluster sizes and concurrency based on usage
5. **Security**: Implement additional security measures as needed











