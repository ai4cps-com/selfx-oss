import os

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]

timezone = "UTC"
enable_utc = True

task_track_started = True
task_ignore_result = False

task_always_eager = True
task_eager_propagates = True

imports = ("selfx.tasks.feature_tasks",)