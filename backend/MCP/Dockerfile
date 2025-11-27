FROM astral/uv:python3.11-trixie-slim AS base

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy your app code
COPY clinic_server.py config.py http_client.py ./
COPY .env* ./

EXPOSE 8000

# Run the app
CMD ["python", "clinic_server.py"]
