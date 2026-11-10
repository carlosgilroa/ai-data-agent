# Running the agent on the Raspberry Pi

The agent runs as one Docker container next to the wallet containers (pipeline scheduler,
Telegram bot, email listener). It reads the wallet lake and the LookML read-only and keeps
its own state (Chroma catalog, SQLite sessions, answer traces) in its own folder.

## Layout on the Pi

```
~/projects/
  wallet/            pipeline + Telegram bot (its own deploy)
  wallet-lookml/     main, mounted read-only at /data/lookml
  ai-data-agent/     main, this repo
    .env             secrets, chmod 600, never in git
    data/            chroma/, sessions.db
    logs/            agent_queries.jsonl
```

The container joins the wallet's `wallet_default` network, so the bot reaches it at
`http://ai-data-agent:8000`. Port 8000 is published on `127.0.0.1` only: the API has no
authentication, so it must never be reachable from the LAN.

## The .env on the Pi

Create `~/projects/ai-data-agent/.env` once, by hand, with `chmod 600`. Use the agent's own
Gemini key (`ai-data-agent-prod`), not the wallet's key. Paths and `DEBUG` are set by
docker-compose.yml, so the file only needs:

```
GEMINI_API_KEY=<the ai-data-agent-prod key>
GEMINI_ROUTER_MODEL=gemini-3.1-flash-lite
GEMINI_SIMPLE_MODEL=gemini-3.5-flash-lite
GEMINI_MODEL=gemini-3.8-flash
MAX_INPUT_TOKENS_PER_QUESTION=40000
```

## Deploying

From the laptop, after the change is merged to main:

```bash
scripts/deploy_pi.sh
```

It clones wallet-lookml and ai-data-agent if missing, pulls main in both (fast-forward only),
rebuilds the image only when `requirements.txt`, `constraints.txt` or the Dockerfile changed, restarts the
container and waits for `/api/v1/health`. The first start downloads the embedding model and
builds the Chroma catalog on the Pi; the Windows index is never copied.

## Fresh data

The agent loads the parquet files into memory and switches DuckDB file access off. Before
each question it checks the files' modification times; when the pipeline has rewritten them
(02:00 daily, or `/sync` in the bot) and they have been untouched for
`DATA_RELOAD_STABLE_SECONDS` (60 s), it loads them into a new in-memory database and swaps it
in. A file that fails to load leaves the previous data in place and is retried on the next
question. LookML changes are picked up the same way and also rebuild the Chroma catalog.

## Cost limits

- Google Cloud: monthly budget on the `ai-data-agent-prod` key's project (20,000 COP).
- Wallet bot: `/ask` only from the allowed chat id, 20 questions a day.
- Agent: 5 steps for simple and 8 for complex questions, and the tool loop stops once a
  question has used `MAX_INPUT_TOKENS_PER_QUESTION` input tokens.

## Checks

```bash
ssh pi 'curl -s http://127.0.0.1:8000/api/v1/health'
ssh pi 'docker logs --tail 50 ai-data-agent'
ssh pi 'docker stats --no-stream ai-data-agent'
```
