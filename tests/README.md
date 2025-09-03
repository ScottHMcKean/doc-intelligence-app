# Document Intelligence Test Suite

This directory contains comprehensive pytest tests for the Document Intelligence application. The tests are designed to be small, isolated integration tests without mocks, following the project's testing philosophy.

## Test Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── run_tests.py                   # Test runner script
├── README.md                      # This file
├── unit/                          # Unit tests
│   └── test_config.py            # Configuration management tests
└── integration/                   # Integration tests
    ├── test_main_app.py          # Main application tests
    ├── test_database_service.py  # Database service tests
    ├── test_agent_service.py     # Agent service tests
    ├── test_document_service.py  # Document service tests
    └── test_storage_service.py   # Storage service tests
```

## Test Philosophy

The tests follow these principles:

1. **No Mocks**: Tests use real integrations when possible, avoiding mocks for better test reliability
2. **Small and Isolated**: Each test focuses on a single component or feature
3. **Integration Focus**: Tests verify that components work together correctly
4. **Fast Execution**: Tests are designed to run quickly using real but lightweight integrations
5. **Comprehensive Coverage**: Tests cover all main components and their interactions

## Running Tests

### Using the Test Runner Script

The `run_tests.py` script provides a convenient way to run tests with various options:

```bash
# Run all tests
./tests/run_tests.py

# Run only unit tests
./tests/run_tests.py --type unit

# Run only integration tests
./tests/run_tests.py --type integration

# Run tests with coverage
./tests/run_tests.py --coverage

# Run tests in verbose mode
./tests/run_tests.py --verbose

# Run specific test file
./tests/run_tests.py --file test_config.py

# Run specific test function
./tests/run_tests.py --function test_app_initialization

# Run tests with specific markers
./tests/run_tests.py --markers "slow"

# Run tests in parallel
./tests/run_tests.py --parallel 4
```

### Using pytest directly

```bash
# Run all tests
uv run pytest tests/

# Run unit tests only
uv run pytest tests/unit/

# Run integration tests only
uv run pytest tests/integration/

# Run with coverage
uv run pytest --cov=src tests/

# Run specific test file
uv run pytest tests/unit/test_config.py

# Run specific test function
uv run pytest -k test_app_initialization
```

## Test Categories

### Unit Tests (`tests/unit/`)

- **Configuration Management**: Tests for `DocConfig`, `DotDict`, and configuration loading
- **Utility Functions**: Tests for helper functions and utilities
- **Data Models**: Tests for data structures and models

### Integration Tests (`tests/integration/`)

- **Main Application**: Tests for the `DocumentIntelligenceApp` class and its orchestration
- **Database Service**: Tests for PostgreSQL operations and vector search
- **Agent Service**: Tests for LLM interactions, embeddings, and RAG capabilities
- **Document Service**: Tests for Databricks job processing
- **Storage Service**: Tests for Databricks volume operations

## Test Fixtures

The `conftest.py` file provides several useful fixtures:

### Configuration Fixtures
- `test_config_data`: Sample configuration data
- `test_config_file`: Temporary configuration file
- `test_config`: Configuration object

### Service Fixtures
- `mock_databricks_client`: Mocked Databricks workspace client
- `test_app_with_mock_client`: Application instance with mocked client

### Data Fixtures
- `sample_document_content`: Sample document content for testing
- `sample_document_filename`: Sample document filename
- `test_user_info`: Test user information
- `test_vector_embedding`: Sample vector embedding
- `test_document_chunks`: Sample document chunks
- `test_conversation_messages`: Sample conversation messages

### Utility Fixtures
- `cleanup_test_files`: Cleanup function for test files

## Test Markers

Tests are automatically marked based on their characteristics:

- `integration`: Tests in the integration directory
- `slow`: Tests that might be slow (database, vector, embedding, llm)
- `requires_databricks`: Tests that need Databricks connection
- `requires_database`: Tests that need database connection

## Test Configuration

The test configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["src/"]
testpaths = ["tests"]
```

## Dependencies

Test dependencies are defined in `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.11.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "ipykernel"
]
```

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`

### Test Structure

```python
class TestComponentName:
    """Test class for ComponentName."""

    def test_specific_functionality(self, fixture1, fixture2):
        """Test specific functionality."""
        # Arrange
        # Act
        # Assert
```

### Best Practices

1. **Use fixtures**: Leverage existing fixtures for common setup
2. **Test one thing**: Each test should verify one specific behavior
3. **Descriptive names**: Test names should clearly describe what is being tested
4. **Clear assertions**: Use specific assertions that clearly indicate what failed
5. **Error handling**: Test both success and failure scenarios

### Example Test

```python
def test_document_upload_success(self, test_config, mock_databricks_client, sample_document_content):
    """Test successful document upload."""
    storage_service = StorageService(client=mock_databricks_client, config=test_config)
    
    success, doc_hash, upload_path, message = storage_service.upload_document(
        file_content=sample_document_content,
        filename="test.txt",
        username="test@databricks.com"
    )
    
    assert success is True
    assert isinstance(doc_hash, str)
    assert upload_path.startswith("/Volumes/")
    assert "successfully" in message.lower()
```

## Continuous Integration

The test suite is designed to work well in CI environments:

1. **Fast execution**: Tests run quickly without external dependencies
2. **Reliable**: Tests use real integrations where possible
3. **Comprehensive**: Full coverage of main components
4. **Parallel execution**: Tests can run in parallel for faster CI

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `src/` is in the Python path
2. **Fixture not found**: Check that fixtures are defined in `conftest.py`
3. **Test failures**: Run tests with `-v` flag for verbose output
4. **Slow tests**: Use markers to skip slow tests during development

### Debug Mode

Run tests with debug output:

```bash
uv run pytest -v -s tests/
```

### Coverage Reports

Generate coverage reports:

```bash
uv run pytest --cov=src --cov-report=html tests/
```

The HTML report will be generated in `htmlcov/index.html`.

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Use appropriate fixtures
3. Add proper docstrings
4. Test both success and failure scenarios
5. Update this README if adding new test categories or fixtures
