FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data/raw/snapshots data/raw/line_history data/processed models logs/plots logs/signals

# Expose web port (Railway auto-detects $PORT)
EXPOSE 5050

# Run web dashboard + background monitor
CMD ["python", "-m", "src.web.serve"]
