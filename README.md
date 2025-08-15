# Document Intelligence App 📄

A comprehensive Streamlit application for document intelligence with Databricks integration. Upload documents, process them with AI, and have intelligent conversations about their content.

## Features ✨

- **Document Upload & Processing**: Upload documents and process them using Databricks `ai_parse_document`
- **Intelligent Chat**: Chat with your documents using Databricks LLM models
- **Conversation History**: Persistent conversation storage with PostgreSQL
- **Vector Search Integration**: Query document databases using vector search
- **Multi-Modal Chat**: Switch between document-specific chat, vector search, and general chat
- **Responsive UI**: Modern, responsive Streamlit interface
- **Databricks Native**: Built specifically for Databricks environments

## Architecture 🏗️

The application follows a modular design with clear separation of concerns:

```
src/doc_intelligence/
├── auth/          # Databricks authentication
├── storage/       # Document upload and volume management
├── processing/    # Serverless job queuing for document processing
├── database/      # PostgreSQL integration and data stores
└── chat/          # LLM integration and chat management
```

## Prerequisites 📋

- Databricks workspace with:
  - Serverless compute enabled
  - Unity Catalog volumes configured
  - LLM serving endpoints (e.g., Llama 3.1)
- Lakebase managed PostgreSQL database
- Python 3.9+ environment

## Quick Start 🚀

### 1. Clone and Setup

```bash
git clone <repository-url>
cd doc-intelligence-app

# Copy environment template
cp environment.template .env
# Edit .env with your configuration
```

### 2. Install Dependencies

```bash
# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### 3. Configure Environment

Edit `.env` file with your settings:

```env
# Databricks Configuration
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-databricks-token

# PostgreSQL Configuration
POSTGRES_HOST=your-postgres-host.com
POSTGRES_USER=your-username
POSTGRES_PASSWORD=your-password

# Volume Paths
DOC_INPUT_VOLUME=/Volumes/main/default/doc_input
DOC_OUTPUT_VOLUME=/Volumes/main/default/doc_output
```

### 4. Setup Databricks Components

1. **Create Volumes**:
   ```sql
   CREATE VOLUME IF NOT EXISTS main.default.doc_input;
   CREATE VOLUME IF NOT EXISTS main.default.doc_output;
   ```

2. **Deploy Processing Notebook**:
   - Create a notebook in your Databricks workspace
   - Use the code from `src/doc_intelligence/processing/job_queue.py:create_processing_notebook()`
   - Save it to `/Workspace/Users/shared/document_processor`

3. **Configure LLM Serving Endpoint**:
   - Set up a serving endpoint for your chosen LLM model
   - Update `LLM_MODEL_NAME` in your `.env` file

### 5. Run the Application

```bash
uv run streamlit run app.py
```

## Usage Guide 🎯

### Document Upload and Processing

1. **Upload Document**: Use the sidebar to upload PDF, DOCX, TXT, or MD files
2. **Processing**: The app automatically:
   - Uploads to Databricks volume
   - Queues serverless job for AI parsing
   - Polls for results
   - Stores chunks in database
3. **Chat**: Once processed, start chatting with your document

### Chat Modes

**General Chat** 💬
- Open-ended conversations with the AI assistant
- No document context

**Document Chat** 📄
- Chat with specific uploaded documents
- AI has access to document content for context
- Automatic chunk retrieval based on your questions

**Vector Search** 🔍
- Query across your entire document database
- Uses vector similarity for relevant content retrieval
- Great for finding information across multiple documents

### Conversation Management

- **History**: All conversations are saved automatically
- **Resume**: Click on any conversation in the sidebar to resume
- **Delete**: Remove conversations you no longer need
- **Titles**: Conversations are automatically titled based on content

## Development 🛠️

### Project Structure

```
doc-intelligence-app/
├── src/doc_intelligence/     # Main application code
│   ├── auth/                # Authentication modules
│   ├── storage/             # Storage and volume operations
│   ├── processing/          # Job queue management
│   ├── database/            # Database integration
│   └── chat/                # LLM and chat functionality
├── app.py                   # Main Streamlit application
├── pyproject.toml          # Project configuration
└── tests/                  # Test suite (when added)
```

### Running Tests

```bash
# Install dev dependencies
uv pip install -e .[dev]

# Run tests
uv run pytest
```

### Code Quality

```bash
# Format code
uv run black src/ app.py

# Sort imports
uv run isort src/ app.py

# Type checking
uv run mypy src/ app.py

# Linting
uv run flake8 src/ app.py
```

## Configuration ⚙️

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABRICKS_HOST` | Databricks workspace URL | Yes |
| `DATABRICKS_TOKEN` | Databricks access token | Yes |
| `POSTGRES_HOST` | PostgreSQL host | Yes |
| `POSTGRES_USER` | PostgreSQL username | Yes |
| `POSTGRES_PASSWORD` | PostgreSQL password | Yes |
| `POSTGRES_DB` | Database name | No (default: doc_intelligence) |
| `LLM_MODEL_NAME` | Databricks LLM model name | No (default: databricks-meta-llama-3-1-70b-instruct) |

### Volume Configuration

Ensure your Databricks volumes are properly configured:
- Input volume for uploaded documents
- Output volume for processed results
- Appropriate permissions for your user/service principal

## Troubleshooting 🔧

### Common Issues

**Authentication Failures**
- Verify Databricks host URL and token
- Check network connectivity
- Ensure token has appropriate permissions

**Document Processing Timeouts**
- Check serverless compute availability
- Verify volume permissions
- Monitor job cluster configuration

**Database Connection Issues**
- Verify PostgreSQL credentials
- Check network security groups
- Ensure database exists

**LLM Errors**
- Verify serving endpoint is running
- Check model name configuration
- Monitor endpoint rate limits

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Submit a pull request

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.