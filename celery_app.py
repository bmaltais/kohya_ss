from celery import Celery

app = Celery(
    "kohya_ss",
    broker="redis://localhost:6379/0",  # Adjust broker URL if needed
    backend="redis://localhost:6379/0",  # Adjust backend URL if needed
    include=["kohya_ss.app.tasks", "kohya_ss.kohya_gui"],
)

# celery -A kohya_ss.celery_app worker -c 1 --loglevel=info
