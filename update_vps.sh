#!/bin/bash

# Ensure we are in the project root
cd "$(dirname "$0")"

# Check if deploy_venv exists
if [ ! -d "deploy_venv" ]; then
    echo "Creating deployment environment..."
    python3 -m venv deploy_venv
    source deploy_venv/bin/activate
    pip install paramiko
else
    source deploy_venv/bin/activate
fi

echo "Starting deployment update..."
python3 /home/morad/Projects/heal-query-hub/scripts/deploy/deploy.py
