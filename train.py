#!/usr/bin/env python3
"""
Entry point for model training in GitHub Actions
"""

import subprocess
import sys
import os

def main():
    # Change to the project directory
    project_dir = "xai-student-suicide-prediction"
    if os.path.exists(project_dir):
        os.chdir(project_dir)

    # Run the training script
    try:
        result = subprocess.run([
            sys.executable,
            "scripts/train_model.py"
        ], check=True, capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
