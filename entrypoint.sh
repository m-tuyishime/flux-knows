#!/usr/bin/env bash
set -e

REPO_DIR="/workspace/flux-knows"

if [ ! -d "$REPO_DIR/.git" ]; then
    echo "Cloning repo"
    git clone https://${GH_TOKEN}@github.com/m-tuyishime/flux-knows.git "$REPO_DIR"
else
    echo "Repo exists, pulling latest changes"
    git -C "$REPO_DIR" pull origin main
fi

if [ -f "$REPO_DIR/requirements.txt" ]; then
    echo "Installing Python dependencies"
    pip install -r "$REPO_DIR/requirements.txt"
else
    echo "No requirements.txt found, skipping dependencies installation"
fi

if [ -n "$HF_TOKEN" ]; then
    echo "Logging into Hugging Face"
    huggingface-cli login --token "$HF_TOKEN"
fi

if [ -n "$PUBLIC_KEY" ]; then
    # Create the .ssh directory if it doesn't exist and set permissions
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    # Add the key to the authorized_keys file
    echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys
    # Set the correct permissions
    chmod 600 ~/.ssh/authorized_keys
    echo "SSH key injected for user 'runpod'."
else
    echo "WARNING: No PUBLIC_KEY environment variable found. SSH access might not be possible."
fi

# Start the SSH server in the foreground to keep the container running
echo "Starting SSH server..."
exec /usr/sbin/sshd -D