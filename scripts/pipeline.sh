#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Check for argument
if [ -z "$1" ]; then
  echo "Usage: ./scripts/run_pipeline.sh <audio_file_path>"
  echo "Example: ./scripts/run_pipeline.sh data/patient_intro.mp3"
  exit 1
fi

# Audio file path is relative to the project root
AUDIO_FILE_PATH="$1"

echo "Ensuring data/output directory exists..."
mkdir -p data/output

# Store the original directory (project root)
ORIGINAL_CWD=$(pwd)

# Change to the backend directory where Pipfile and Pipfile.lock are located
# This ensures 'pipenv run' finds the correct virtual environment
cd backend

# Convert the audio file path to be absolute, so it's correct regardless of CWD change
ABS_AUDIO_FILE_PATH=$(realpath "$ORIGINAL_CWD/$AUDIO_FILE_RELATIVE_PATH")

pipenv run python -c "
import asyncio
import sys
import os

try:
    from pipeline import run_pipeline_internal
    # Run the async pipeline function
    success = asyncio.run(run_pipeline_internal('$AUDIO_FILE_PATH'))
    if not success:
        sys.exit(1) # Exit with error if pipeline failed
except ImportError as e:
    print(f'Error importing pipeline: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'An unexpected error occurred during script execution: {e}', file=sys.stderr)
    sys.exit(1)
"

# Change back to the original directory (project root)
cd "$ORIGINAL_CWD"

echo "Pipeline execution finished."
