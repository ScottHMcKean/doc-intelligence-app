#!/usr/bin/env python3
"""Deployment script for Databricks Asset Bundle."""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd: list, cwd: str = None) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def build_wheel() -> bool:
    """Build the Python wheel package."""
    print("Building Python wheel...")
    success, output = run_command(["uv", "build"])
    if success:
        print("✅ Wheel built successfully")
        return True
    else:
        print(f"❌ Failed to build wheel: {output}")
        return False


def deploy_bundle(target: str = "dev") -> bool:
    """Deploy the bundle to Databricks."""
    print(f"Deploying bundle to target: {target}")
    success, output = run_command(
        ["databricks", "bundle", "deploy", "--target", target]
    )
    if success:
        print(f"✅ Bundle deployed successfully to {target}")
        return True
    else:
        print(f"❌ Failed to deploy bundle: {output}")
        return False


def validate_bundle() -> bool:
    """Validate the bundle configuration."""
    print("Validating bundle configuration...")
    success, output = run_command(["databricks", "bundle", "validate"])
    if success:
        print("✅ Bundle validation successful")
        return True
    else:
        print(f"❌ Bundle validation failed: {output}")
        return False


def run_job(job_name: str, target: str = "dev", parameters: dict = None) -> bool:
    """Run a specific job."""
    print(f"Running job: {job_name}")

    cmd = ["databricks", "bundle", "run", job_name, "--target", target]
    if parameters:
        for key, value in parameters.items():
            cmd.extend(["--parameter", f"{key}={value}"])

    success, output = run_command(cmd)
    if success:
        print(f"✅ Job {job_name} started successfully")
        return True
    else:
        print(f"❌ Failed to run job {job_name}: {output}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Deploy and manage Databricks Asset Bundle"
    )
    parser.add_argument(
        "--target", default="dev", choices=["dev", "prod"], help="Target environment"
    )
    parser.add_argument(
        "--action",
        required=True,
        choices=["validate", "build", "deploy", "run-job"],
        help="Action to perform",
    )
    parser.add_argument("--job-name", help="Job name (required for run-job action)")
    parser.add_argument(
        "--parameters", nargs="*", help="Job parameters in key=value format"
    )

    args = parser.parse_args()

    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    success = False

    if args.action == "validate":
        success = validate_bundle()
    elif args.action == "build":
        success = build_wheel()
    elif args.action == "deploy":
        if not build_wheel():
            sys.exit(1)
        success = deploy_bundle(args.target)
    elif args.action == "run-job":
        if not args.job_name:
            print("❌ Job name is required for run-job action")
            sys.exit(1)

        # Parse parameters
        params = {}
        if args.parameters:
            for param in args.parameters:
                if "=" in param:
                    key, value = param.split("=", 1)
                    params[key] = value

        success = run_job(args.job_name, args.target, params)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()











