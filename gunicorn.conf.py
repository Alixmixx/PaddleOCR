# gunicorn.conf.py

import multiprocessing
import os

bind = '0.0.0.0:8000'  # Address and port to bind the server
workers = 4 # multiprocessing.cpu_count() * 2 + 1  # Number of worker processes
worker_class = 'uvicorn.workers.UvicornWorker'  # Worker class to use
worker_connections = 1000  # For async workers
timeout = 60  # Workers silent for more than this many seconds are killed and restarted
keepalive = 2  # The number of seconds to wait for the next request on a Keep-Alive HTTP connection
errorlog = '-'  # Redirect error logs to stderr
accesslog = '-'  # Redirect access logs to stdout
loglevel = 'info'  # Logging level
capture_output = True  # Capture stdout/stderr output for logs
graceful_timeout = 30  # Timeout for graceful workers restart
preload_app = True  # Load application code before the worker processes are forked

# Environment variables
raw_env = [
    'PADDLE_OCR_API_KEY=' + os.environ.get('PADDLE_OCR_API_KEY', 'mirinae'),
]
