FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Drop root after the image filesystem is assembled
RUN addgroup --system app && adduser --system --ingroup app app

# Create data directories (all under data/ so Railway volume at /app/data persists everything)
RUN mkdir -p data/raw/snapshots data/raw/line_history data/processed data/logs/plots data/logs/signals models \
    && chmod +x entrypoint.sh \
    && chown -R app:app /app

# Expose web port (Railway auto-detects $PORT)
ENV MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 5050

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-5050}/healthz')" || exit 1

# Entrypoint migrates old file paths then starts the app
USER app
CMD ["./entrypoint.sh"]
