# AI data agent: FastAPI service, built for the Raspberry Pi (linux/arm64) and also runs on amd64.
# Python 3.13 and constraints.txt match the dev venv the tests run in.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Run as the Pi's own user id so files written to bind mounts (Chroma, sessions, logs) belong to it
ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd -g ${APP_GID} agent && useradd -m -u ${APP_UID} -g ${APP_GID} agent

WORKDIR /app

COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

COPY app/ /app/app/
COPY agent_instructions.yaml /app/agent_instructions.yaml
RUN mkdir -p /app/data /app/logs /home/agent/.cache && chown -R agent:agent /app /home/agent/.cache

USER agent
EXPOSE 8000

# One worker: a single writer for the SQLite session store and one in-memory copy of the data
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
