#!/bin/bash

# Variables
# Load variables from .env file in the current directory
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found in the current directory."
    exit 1
fi

# Sync project files to remote server
rsync -avz --progress --stats --exclude results/ --exclude samples/ --exclude tmp/ --exclude .git/ -e "ssh -p $PORT -i $SSH_KEY_PATH" ./ $USER@$IP:$TARGET_VOLUME/DiT-optimizations-benchmark