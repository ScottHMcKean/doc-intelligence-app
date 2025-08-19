# Document Intelligence App 📄

A comprehensive document intelligence application with Databricks integration, built with a modular, service-oriented architecture. Upload documents, process them with AI, and have intelligent conversations about their content using RAG (Retrieval Augmented Generation).

## Features ✨

- **Document Upload & Processing**: Upload documents and process them using Databricks AI parsing
- **Intelligent Chat**: Chat with your documents using Databricks LLM models
- **Vector Search**: Semantic search across document chunks using pgvector
- **Conversation Management**: Persistent conversation storage with PostgreSQL
- **Multi-Modal Chat**: Switch between document-specific chat and general chat
- **Databricks Native**: Built specifically for Databricks environments with Unity Catalog

## Architecture Overview 🏗️

The application follows a clean, modular design with well-defined service boundaries:

```
src/doc_intelligence/
├── services/          # Core service layer
│   ├── storage_service.py      # Databricks volume management
│   ├── document_service.py     # Document processing orchestration
│   ├── database_service.py     # PostgreSQL integration
│   └── agent_service.py        # LLM + RAG capabilities
├── workflows/         # Business logic orchestration
│   ├── document_workflow.py    # End-to-end document processing
│   └── conversation_workflow.py # Chat and conversation management
├── database/          # Database schema and models
│   └── schema.py              # SQLAlchemy models with pgvector support
├── storage/           # Storage abstractions
│   └── volume_storage.py      # Databricks volume operations
├── agent/             # Agent and conversation management
│   ├── conversation_manager.py # Chat state management
│   ├── rag_workflow.py        # RAG pipeline implementation
│   └── checkpointing.py       # Conversation persistence
└── config.py          # Configuration management
```

## Core Services 🔧

### **StorageService** 🗄️
- Manages Databricks Unity Catalog volumes for document storage
- Handles file upload/download operations
- Provides volume path management and file operations

### **DocumentService** 📄
- Orchestrates document processing workflows
- Manages document metadata and status tracking
- Handles document chunking and processing state

### **DatabaseService** 💾
- PostgreSQL integration using the same connection pattern as your `ai_parse.ipynb`
- Manages users, conversations, documents, and message history
- Handles database schema creation and migrations
- Uses dynamic credential generation via Databricks SDK

### **AgentService** 🤖
- Combines Databricks LLM capabilities with RAG
- Manages conversation state using LangGraph
- Provides vector search across document chunks
- Handles embedding generation and similarity search

## Workflows 🔄

### **DocumentWorkflow**
Coordinates the end-to-end document processing:
1. **Upload**: Store document in Databricks volume
2. **Process**: Queue AI parsing job
3. **Chunk**: Split document into searchable chunks
4. **Embed**: Generate vector embeddings
5. **Store**: Save chunks and metadata to database

### **ConversationWorkflow**
Manages intelligent conversations:
1. **Context**: Retrieve relevant document chunks using vector search
2. **Generate**: Use LLM to create contextual responses
3. **Persist**: Save conversation history and state
4. **Manage**: Handle conversation titles and metadata

## Configuration ⚙️

The application uses a centralized `config.yaml` file for all configuration:

```yaml
# Databricks Configuration
databricks:
  host: "https://your-workspace.cloud.databricks.com"
  token: "your-databricks-token"

# Database Configuration
database:
  instance_name: "shm"
  host: "your-database-host.database.azuredatabricks.net"
  port: 5432
  database: "databricks_postgres"
  user: "your-username"

# Agent Configuration
agent:
  embedding_endpoint: "databricks-gte-large"
  llm:
    endpoint: "databricks-claude-sonnet-4"
    max_tokens: 512
    temperature: 0.1
```

**No environment variables needed** - everything is configured through the YAML file.

## Quick Start 🚀

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

## Database Schema 🗃️

The application uses PostgreSQL with pgvector extension:

- **Users**: User management with Databricks user ID mapping
- **Documents**: Document metadata and processing status
- **Document Chunks**: Text chunks with vector embeddings
- **Conversations**: Chat sessions and thread management
- **Messages**: Individual chat messages with metadata

## Key Design Principles 🎯

### **Modularity**
- Each service has a single, well-defined responsibility
- Services communicate through clean interfaces
- Easy to test and maintain individual components

### **Graceful Degradation**
- App remains functional even when external services are unavailable
- Automatic fallbacks for missing capabilities
- Clear status reporting for service availability

### **Databricks Integration**
- Uses Databricks SDK for authentication and database access
- Dynamic credential generation for database connections
- Unity Catalog volume management for document storage

### **Vector Search**
- pgvector integration for semantic document search
- Automatic chunking and embedding generation
- Efficient similarity search across document database

## Development 🛠️

### Project Structure
```
doc-intelligence-app/
├── src/doc_intelligence/     # Main application code
├── app.py                    # Streamlit application entry point
├── config.yaml              # Configuration file
├── pyproject.toml           # Project dependencies
└── tests/                   # Test suite
```

### Running Tests
```bash
uv run pytest
```

### Code Quality
```bash
uv run black src/ app.py
uv run isort src/ app.py
uv run mypy src/ app.py
```

## Troubleshooting 🔧

### Common Issues

**Database Connection Errors**
- Verify database instance name in config
- Check Databricks workspace permissions
- Ensure database instance is accessible

**LLM/Embedding Errors**
- Verify endpoint names in config
- Check endpoint status in Databricks
- Ensure proper authentication

**Document Processing Issues**
- Check volume permissions
- Verify job cluster configuration
- Monitor processing notebook execution

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Follow the modular design principles
4. Add tests for new functionality
5. Submit a pull request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.