# Multi-Agent System Dockerfile
# Build a containerized environment for the multi-agent system

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY multi_agent_system/ ./multi_agent_system/
COPY outputs/ ./outputs/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port for future API service
EXPOSE 8000

# Default command: Run the test suite
CMD ["python", "multi_agent_system/test_runner.py"]

# Alternative commands (uncomment to use):
# Interactive CLI: CMD ["python", "multi_agent_system/cli.py"]
# Python shell: CMD ["python"]
