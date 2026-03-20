FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Drop root after the image filesystem is assembled
RUN addgroup --system app && adduser --system --ingroup app app

# Create default app directories; hosted log/cache persistence is resolved at runtime.
RUN mkdir -p data/raw/snapshots data/raw/line_history data/processed data/logs/plots data/logs/signals models \
    && chmod +x entrypoint.sh \
    && chown -R app:app /app

# Expose web port (Railway auto-detects $PORT)
ENV MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 5050

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-5050}/healthz')" || exit 1

# Entrypoint fixes mounted-volume permissions, then drops to the app user.
USER root
CMD ["./entrypoint.sh"]
