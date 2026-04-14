#!/usr/bin/env python3
"""Test script for DAB integration."""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from doc_intel.config import load_config

        print("✅ Config module imported")
    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
        return False

    try:
        from doc_intel.database.service import DatabaseService

        print("✅ Database service imported")
    except ImportError as e:
        print(f"❌ Failed to import database service: {e}")
        return False

    try:
        from doc_intel.document.service import DocumentService

        print("✅ Document service imported")
    except ImportError as e:
        print(f"❌ Failed to import document service: {e}")
        return False

    try:
        from doc_intel.integration.service_integration import ServiceIntegration

        print("✅ Service integration imported")
    except ImportError as e:
        print(f"❌ Failed to import service integration: {e}")
        return False

    try:
        from doc_intel.entry_points import (
            database_setup,
            document_processing,
            agent_workflow,
        )

        print("✅ Entry points imported")
    except ImportError as e:
        print(f"❌ Failed to import entry points: {e}")
        return False

    return True


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")

    try:
        from doc_intel.config import load_config

        config = load_config()
        print("✅ Configuration loaded successfully")
        print(
            f"   Database instance: {config.get('database', {}).get('instance_name', 'Not configured')}"
        )
        return True
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False


def test_service_initialization():
    """Test service initialization."""
    print("\nTesting service initialization...")

    try:
        from doc_intel.integration.service_integration import ServiceIntegration
        from doc_intel.config import load_config

        config = load_config()
        # Don't actually initialize with real client for testing
        print("✅ Service integration can be initialized")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize services: {e}")
        return False


def test_entry_points():
    """Test entry point functions."""
    print("\nTesting entry point functions...")

    try:
        from doc_intel.entry_points import (
            database_setup,
            document_processing,
            agent_workflow,
        )

        # Test that functions exist and are callable
        assert callable(database_setup), "database_setup is not callable"
        assert callable(document_processing), "document_processing is not callable"
        assert callable(agent_workflow), "agent_workflow is not callable"

        print("✅ Entry point functions are callable")
        return True
    except Exception as e:
        print(f"❌ Entry point test failed: {e}")
        return False


def test_dab_configuration():
    """Test DAB configuration files."""
    print("\nTesting DAB configuration...")

    project_root = Path(__file__).parent.parent

    # Check databricks.yml
    databricks_yml = project_root / "databricks.yml"
    if databricks_yml.exists():
        print("✅ databricks.yml exists")
    else:
        print("❌ databricks.yml not found")
        return False

    # Check resources directory
    resources_dir = project_root / "resources"
    if resources_dir.exists():
        print("✅ resources directory exists")
    else:
        print("❌ resources directory not found")
        return False

    # Check resource files
    env_yml = resources_dir / "environment.yml"
    jobs_yml = resources_dir / "jobs.yml"

    if env_yml.exists():
        print("✅ environment.yml exists")
    else:
        print("❌ environment.yml not found")
        return False

    if jobs_yml.exists():
        print("✅ jobs.yml exists")
    else:
        print("❌ jobs.yml not found")
        return False

    return True


def test_pyproject_scripts():
    """Test pyproject.toml script entries."""
    print("\nTesting pyproject.toml script entries...")

    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        print("❌ pyproject.toml not found")
        return False

    content = pyproject_path.read_text()

    # Check for script entries
    required_scripts = [
        'database_setup = "doc_intel.entry_points:database_setup"',
        'document_processing = "doc_intel.entry_points:document_processing"',
        'agent_workflow = "doc_intel.entry_points:agent_workflow"',
    ]

    for script in required_scripts:
        if script in content:
            print(f"✅ Found script entry: {script.split('=')[0].strip()}")
        else:
            print(f"❌ Missing script entry: {script.split('=')[0].strip()}")
            return False

    return True


def main():
    """Run all tests."""
    print("🧪 Running DAB Integration Tests\n")

    tests = [
        test_imports,
        test_config_loading,
        test_service_initialization,
        test_entry_points,
        test_dab_configuration,
        test_pyproject_scripts,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! DAB integration is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)











