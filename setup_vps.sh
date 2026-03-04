#!/bin/bash

# Stop on error
set -e

export DEBIAN_FRONTEND=noninteractive

echo "Updating system packages..."
apt-get update && apt-get upgrade -y

echo "Installing prerequisites..."
apt-get install -y ca-certificates curl gnupg lsb-release unzip

echo "Installing Docker..."
# Add Docker's official GPG key
mkdir -p /etc/apt/keyrings
if [ -f /etc/apt/keyrings/docker.gpg ]; then
    rm /etc/apt/keyrings/docker.gpg
fi
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

echo "Installing Docker Compose (standalone)..."
curl -SL https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

echo "Docker installed successfully!"
docker --version
docker-compose --version
