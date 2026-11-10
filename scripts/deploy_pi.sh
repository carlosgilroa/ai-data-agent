#!/usr/bin/env bash
# Deploy the AI data agent to the Raspberry Pi from this machine (Git Bash on Windows works).
#
#   scripts/deploy_pi.sh
#
# On the Pi it clones wallet-lookml and ai-data-agent into ~/projects if they are missing, pulls main
# in both (fast-forward only, so local changes on the Pi stop the deploy instead of being overwritten),
# rebuilds the image only when requirements.txt or the Dockerfile changed, restarts the agent and
# waits for its health check. The wallet repo is not touched: it has its own deploy.
# The agent's .env must already exist on the Pi (see docs/deployment/raspberry_pi.md).
set -euo pipefail

PI_HOST="${PI_HOST:-pi}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_claude_pi}"

ssh -i "$SSH_KEY" -o IdentitiesOnly=yes "$PI_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
cd ~/projects

for repo in wallet-lookml ai-data-agent; do
  if [ ! -d "$repo/.git" ]; then
    echo "== Cloning $repo"
    git clone "git@github.com:carlosgilroa/$repo.git" "$repo"
  fi
done

echo "== wallet-lookml"
git -C wallet-lookml checkout -q main
git -C wallet-lookml pull --ff-only -q origin main
git -C wallet-lookml log --oneline -1

echo "== ai-data-agent"
cd ai-data-agent
git checkout -q main
before=$(git rev-parse HEAD)
git pull --ff-only -q origin main
git log --oneline -1

if [ ! -f .env ]; then
  echo "Missing ~/projects/ai-data-agent/.env; create it first (docs/deployment/raspberry_pi.md)." >&2
  exit 1
fi
chmod 600 .env
mkdir -p data logs

if ! docker image inspect ai-data-agent >/dev/null 2>&1 || ! git diff --quiet "$before" HEAD -- requirements.txt Dockerfile; then
  echo "== Building the image (first time on a Pi takes several minutes)"
  docker compose build
fi
docker compose up -d
docker compose restart ai-data-agent

echo "== Waiting for the health check (the first start builds the Chroma catalog)"
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1; then
    curl -fsS http://127.0.0.1:8000/api/v1/health; echo
    exit 0
  fi
  sleep 5
done
echo "The agent did not become healthy within 5 minutes; check: docker logs ai-data-agent" >&2
exit 1
REMOTE
