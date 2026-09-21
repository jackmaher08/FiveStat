"""Production Gunicorn configuration for Railway."""

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
worker_class = "sync"
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

# Recycle the worker periodically to limit long-lived memory growth from
# pandas/matplotlib workloads without multiplying baseline memory usage.
max_requests = 500
max_requests_jitter = 50

accesslog = "-"
errorlog = "-"
capture_output = True
loglevel = os.environ.get("LOG_LEVEL", "info")
preload_app = False
