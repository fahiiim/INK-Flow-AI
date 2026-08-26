# Production Deployment on the Shared EC2 Instance

The FastAPI service runs in its own container under
`/opt/tattoo-hysteria-ai`. It joins the backend's existing external Docker
network and does not publish port 8001 to the EC2 host.

## 1. EC2 prerequisites

Docker and the Docker Compose plugin should already exist because the Django
stack uses them. Confirm that the `ubuntu` user can run Docker:

```bash
docker --version
docker compose version
```

Create the dedicated folder and shared network:

```bash
sudo mkdir -p /opt/tattoo-hysteria-ai
sudo chown -R ubuntu:ubuntu /opt/tattoo-hysteria-ai
sudo chmod -R 755 /opt/tattoo-hysteria-ai

docker network inspect tattoo_hysteria_net >/dev/null 2>&1 \
  || docker network create tattoo_hysteria_net
```

Clone the repository. The final dot is required:

```bash
cd /opt/tattoo-hysteria-ai
git clone https://github.com/fahiiim/INK-Flow-AI.git .
```

For a private repository, use a separate read-only GitHub deploy key on the
EC2 instance. Do not reuse the GitHub Actions to EC2 SSH key.

## 2. Production environment file

GitHub Actions creates the server-only environment file automatically during
every production deployment. Do not create it manually and do not commit it.

The workflow generates this file at `/opt/tattoo-hysteria-ai/.env`:

```env
OPENAI_API_KEY=value_from_the_github_production_secret
OPENAI_TEMPERATURE=0.0
OPENAI_TIMEOUT_SECONDS=30
OPENAI_MAX_RETRIES=2
```

The workflow transfers the key through encrypted SSH, writes a temporary file
with restrictive permissions, and atomically replaces `.env`. It does not print
the key. The Compose file loads it only at container runtime.

The service does not send Telegram messages itself, so it does not need a
Telegram bot token. It returns a Telegram-ready summary to Django.

## 3. First manual deployment

Run the same scoped script used by CI/CD:

```bash
cd /opt/tattoo-hysteria-ai
bash scripts/deploy-production.sh
```

The script performs these operations only for the AI service:

- Validates the production Compose configuration.
- Creates the shared network only when it does not exist.
- Builds the AI image from the current checkout.
- Starts `tattoo_hysteria_ai` without publishing port 8001.
- Waits for the readiness health check.
- Keeps the prior AI image under `tattoo-hysteria-ai:rollback`.

It does not remove networks, volumes, backend containers, or global images.

## 4. GitHub Actions production environment

In GitHub, open `Settings`, then `Environments`, and create an environment
named `production`. Add a required reviewer if your GitHub plan supports it.

Add these environment secrets:

- `EC2_HOST`: Public EC2 IP address or DNS name.
- `EC2_USER`: Usually `ubuntu`.
- `EC2_SSH_PORT`: Usually `22`.
- `EC2_SSH_PRIVATE_KEY`: Private deployment key used only by Actions.
- `EC2_SSH_KNOWN_HOSTS`: Verified SSH known-hosts entry for the EC2 host.
- `OPENAI_API_KEY`: Project-scoped production OpenAI API key.

Generate a dedicated Actions deployment key on a trusted machine:

```bash
ssh-keygen -t ed25519 \
  -C "github-actions-tattoo-hysteria-ai" \
  -f github-actions-tattoo-hysteria-ai
```

Add the public key to `/home/ubuntu/.ssh/authorized_keys` on EC2. Store the
entire private key in `EC2_SSH_PRIVATE_KEY`.

Generate the known-hosts line only after verifying the EC2 SSH host-key
fingerprint through a trusted channel:

```bash
ssh-keyscan -H YOUR_EC2_HOST
```

Store the complete output line in `EC2_SSH_KNOWN_HOSTS`.

The workflow does not need AWS access keys, database credentials, Django
secrets, or Telegram credentials. GitHub's automatic `GITHUB_TOKEN` is
sufficient for checkout.

## 5. CI/CD behavior

Pull requests to `main` run the complete tests and validate the Docker build.
A push to `main`, or a manual workflow dispatch from `main`, additionally:

1. Connects to EC2 over verified SSH.
2. Creates the protected production `.env` from the GitHub environment secret.
3. Refuses to overwrite tracked local changes in the deployment checkout.
4. Fast-forwards the checkout to the exact workflow commit.
5. Runs `scripts/deploy-production.sh`.
6. Rolls back to the previous AI image if the new container is unhealthy.

The EC2 checkout must already be able to read the GitHub repository. A public
repository needs no Git credential. A private repository should use an EC2-side
read-only deploy key.

## 6. Backend connectivity checks

Enter the Django container:

```bash
cd /opt/tattoo-hysteria-backend
docker compose -f docker-compose.prod.yml exec backend sh
```

Verify private service discovery and liveness:

```bash
curl -fsS http://tattoo_hysteria_ai:8001/health
```

Verify the analysis endpoint with a proper POST body:

```bash
curl -fsS -X POST \
  http://tattoo_hysteria_ai:8001/api/v1/inquiries/analyze \
  -H 'Content-Type: application/json' \
  -d '{
    "current_message": "I want a 10cm floral tattoo on my arm.",
    "new_image_urls": [],
    "existing_db_state": {},
    "recent_chat_history": []
  }'
```

Leave the backend container with `exit`.

Confirm that the AI service has no published host port:

```bash
docker port tattoo_hysteria_ai
```

The command should print nothing. Do not add EC2 security-group access for
port 8001.

## 7. Operations

Check status and logs:

```bash
cd /opt/tattoo-hysteria-ai
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs --tail 200 ai
```

Manually restore the retained AI rollback image if necessary:

```bash
cd /opt/tattoo-hysteria-ai
docker image tag tattoo-hysteria-ai:rollback tattoo-hysteria-ai:latest
docker compose -f docker-compose.prod.yml up -d --force-recreate
```

Never run `docker system prune -af` on this shared EC2 instance.

## 8. Required learning-layer follow-up

The two current backend endpoints analyze inquiries and create Telegram-ready
summaries. They do not send `StudioLearningRecord` objects back to the AI
service. The default FastAPI dependency also does not load a persisted FAISS
index at startup.

This means the deployed service will safely remain in cold-start mode until a
separate backend-to-AI learning contract is agreed and implemented. Before
enabling automatic artist or price suggestions, define one of these flows:

- An internal endpoint that accepts a validated learning record, embeds it,
  updates FAISS, and persists the index atomically.
- A startup synchronization job that reads verified records through a backend
  interface and rebuilds or restores the FAISS index.

Do not bypass the ten-record cold-start threshold. Keep one Uvicorn worker until
FAISS writes and persistence are coordinated across workers or replicas.

When validating Compose, use the quiet form below because the normal `config`
output expands environment values and can expose secrets in logs:

```bash
docker compose -f docker-compose.prod.yml config --quiet
```
