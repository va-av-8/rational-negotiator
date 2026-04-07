FROM python:3.11-slim

# Install curl for healthchecks and uv
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install uv

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml .

# Install dependencies with uv
RUN uv pip install --system -e .

# Copy application files
COPY main.py .
COPY negotiator.py .

# Expose port
ENV PORT=8080
EXPOSE 8080

# ENTRYPOINT accepts CLI args from compose; CMD provides defaults
ENTRYPOINT ["python", "main.py"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
