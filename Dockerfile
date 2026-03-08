FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data/raw/snapshots data/raw/line_history data/processed models logs/plots logs/signals

# Default: run the monitor (continuous mode)
CMD ["python", "-m", "src.bot", "monitor", "--interval", "6"]
