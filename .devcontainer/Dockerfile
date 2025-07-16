# Use your base image
FROM nvcr.io/nvidia/pytorch:23.08-py3

# Install SSH client if not already installed
RUN apt-get update && apt-get install -y openssh-client && rm -rf /var/lib/apt/lists/*

# Create SSH directory with correct permissions
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Add known_hosts to avoid SSH warnings (optional)
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

# Set environment variable for SSH agent socket (matches devcontainer.json)
ENV SSH_AUTH_SOCK=/ssh-agent

# Entry point or CMD remains as is (usually your devcontainer handles this)
