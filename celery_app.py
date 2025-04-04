# celery_app.py
from celery import Celery
import os
import sys

# Ensure src is in path for task discovery
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Configure Celery
# Use Redis as the message broker and result backend
# Assumes Redis is running on localhost:6379
redis_url = 'redis://localhost:6379/0'

celery = Celery(
    'ml_agent_tasks', # Name of the Celery application
    broker=redis_url,
    backend=redis_url,
    include=['tasks'] # List of modules where tasks are defined
)

# Optional Celery configuration
celery.conf.update(
    result_expires=3600, # Keep task results for 1 hour
    task_serializer='json',
    result_serializer='json',
    accept_content=['json']
)

if __name__ == '__main__':
    # This allows running the worker from the command line:
    # celery -A celery_app worker --loglevel=info
    celery.start() 