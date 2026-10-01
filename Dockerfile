FROM python:3.8-slim-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The Flask runtime has its own dependency file because the repository also
# contains optional notebook and audit dependencies.
COPY ["Flask Deployed App/requirements.txt", "/tmp/requirements.txt"]
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY ["Flask Deployed App", "/app/Flask Deployed App"]
COPY ["Model", "/app/Model"]

WORKDIR "/app/Flask Deployed App"

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]
