FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data directories (all under data/ so Railway volume at /app/data persists everything)
RUN mkdir -p data/raw/snapshots data/raw/line_history data/processed data/logs/plots data/logs/signals models

# Expose web port (Railway auto-detects $PORT)
EXPOSE 5050

# Entrypoint migrates old file paths then starts the app
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]
