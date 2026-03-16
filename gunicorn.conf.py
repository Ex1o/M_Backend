bind = "0.0.0.0:10000"
worker_class = "gthread"
workers = 1
threads = 4
timeout = 600
graceful_timeout = 120
keepalive = 65

# Avoid writing large worker temp files to slower disk where possible.
worker_tmp_dir = "/dev/shm"
