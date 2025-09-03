#!/usr/bin/env python3
"""Test runner script for Document Intelligence application tests."""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run Document Intelligence tests")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--markers", help="Pytest markers to run (e.g., 'slow', 'requires_databricks')"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        help="Run tests in parallel with specified number of workers",
    )
    parser.add_argument("--file", help="Run specific test file")
    parser.add_argument("--function", help="Run specific test function")

    args = parser.parse_args()

    # Change to project root directory
    project_root = Path(__file__).parent.parent
    print(f"Project root: {project_root}")

    # Build pytest command
    cmd = ["uv", "run", "pytest"]

    # Add test path based on type
    if args.type == "unit":
        cmd.append("tests/unit/")
    elif args.type == "integration":
        cmd.append("tests/integration/")
    elif args.type == "all":
        cmd.append("tests/")

    # Add specific file if provided
    if args.file:
        cmd = ["uv", "run", "pytest", f"tests/{args.type}/{args.file}"]

    # Add specific function if provided
    if args.function:
        cmd.extend(["-k", args.function])

    # Add markers if provided
    if args.markers:
        cmd.extend(["-m", args.markers])

    # Add verbose flag
    if args.verbose:
        cmd.append("-v")

    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])

    # Add parallel execution if requested
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])

    # Add other useful pytest options
    cmd.extend(
        [
            "--tb=short",  # Shorter traceback format
            "--strict-markers",  # Strict marker checking
            "--disable-warnings",  # Disable warnings for cleaner output
        ]
    )

    # Run the tests
    success = run_command(cmd, f"Running {args.type} tests")

    if success:
        print(f"\n🎉 All {args.type} tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print(f"\n💥 Some {args.type} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
