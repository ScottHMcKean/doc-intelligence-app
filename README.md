# Doc Intelligence on Databricks

This repository is a comprehensive document intelligence applicaton built on top of Databricks. It allows users to upload and process documents, and then have intelligent conversations with the documents they uploaded, or a global vector search database. See our blog for more information.

## Buisness Problem

Businesses waste valuable time searching through scattered documents for answers. Generic AI assistants can’t provide precise, document-specific insights or track how information is used. Productivity soars when users can chat directly with their own documents, ask targeted questions, and get context-aware answers.

But to truly support enterprise needs, a system must do more: track conversations for compliance and continuity, let users reference specific documents, and provide visibility into how documents are accessed and used. This platform delivers those capabilities—enabling smarter, faster work with full traceability and control.

## Solution

Databricks is the ideal platform for document intelligence because it delivers everything needed—secure storage, scalable compute, LLM endpoints, and vector search—in one place. You can deploy the full solution as code, with integrated governance via Unity Catalog for access control and auditability. Databricks ensures high performance for document processing and chat, while keeping data secure and compliant from end to end.

This repository provides a complete solution with modular services for running a scalable and secure document intelligence solution.

## Application Flow

To use the applicaton users authenticate via the Databricks workspace client, establishing a session and verifying their database record. They can view or resume past conversations, or start new ones, with all chat history and context managed in the database.

Documents are uploaded to Unity Catalog volumes and processed asynchronously for chunking and embedding. Vector search retrieves relevant content for RAG (Retrieval Augmented Generation) with Databricks LLMs, and users can toggle between searching all documents or just those in the current session.

## Services

This repository assembles a set of modular services that are needed to run a document intelligence solution:

1. Storage Service (Volumes)
2. Database Service (Lakebase)
3. Document Service (DBSQL)
4. Agent Service (Serving)

These four services are brought together by a front end that is delivered with Databricks Apps.

### Stroage Service

The Storage Service is responsible for managing document storage and retrieval.. It handles file upload/download operations, provides volume path management and file operations.

### Database Service

The Database Service is responsible for managing the PostgreSQL database for the document intelligence solution. It handles database schema creation and migrations, and provides dynamic credential generation via Databricks SDK. We use pgvector extension for vector similarity search on small batches of documents and Databricks Vector Search for an overall repository. See [data_model.md](data_model.md) for a detailed description of the database schema.

### Document Service

The Document Service is responsible for orchestrating the document processing workflows. It manages document metadata and status tracking, and handles document chunking and processing state.

### Agent Service

The Agent Service is responsible for combining Databricks LLM capabilities with RAG. It manages conversation state using LangGraph, provides vector search across document chunks, and handles embedding generation and similarity search.


## TODO

- [ ] Write blog



## Development

### 1. Setup

```bash
git clone <repository-url>
cd doc-intelligence-app

# Install dependencies
uv sync --dev
```

### 2. Configure

Edit `config.yaml` with your:
- Databricks workspace details
- Database instance information
- LLM and embedding endpoints

### 3. Run

```bash
uv run streamlit run app.py
```

## Development

This application uses uv for package management and is designed to work with Serverless v3 on Databricks. We use pytest for testing.

### Running Tests
```bash
uv run pytest
```

### Troubleshooting FAQ

**Q: I'm getting database connection errors. What should I check?**  
A:  
First, verify that the database instance name in your configuration is correct. Next, check that your Databricks workspace permissions are properly set. Finally, ensure that the database instance is accessible from your environment.

**Q: Why am I seeing errors with LLM or embedding endpoints?**  
A:  
Begin by confirming that the endpoint names in your configuration are correct. Then, check the status of the endpoints in Databricks. Also, make sure you have set up proper authentication.

**Q: Document processing isn't working. What could be wrong?**  
A:  
Check that you have the correct permissions for the storage volume. Then, verify that the job cluster configuration is correct. Finally, monitor the execution of the processing notebook for any errors or failures.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the modular design principles
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.