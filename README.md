# INK Flow AI

INK Flow AI is a small FastAPI service for analyzing tattoo inquiries. It uses
the latest message, up to 30 recent chat messages, saved inquiry details, and
optional reference images to suggest an artist, classify risk, and draft a
reply.

## Requirements

- Python 3.12
- An OpenAI API key

## Run locally

Open PowerShell in the project folder and create a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Create a `.env` file in the project folder:

```env
OPENAI_API_KEY=your_api_key_here
```

Start the API with:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8001 --reload
```

The interactive API documentation is available at
http://127.0.0.1:8001/docs.

## Using the API

Send a `POST` request to `/api/v1/inquiries/analyze` with a payload like this:

```json
{
  "current_message": "I want a 10cm black floral tattoo on my arm.",
  "new_image_urls": [],
  "existing_db_state": {},
  "recent_chat_history": [
    {"role": "user", "content": "I am thinking about a floral design."}
  ]
}
```

Use `/api/v1/inquiries/telegram-summary` for high-risk inquiries. It returns a
staff summary together with the generated draft reply.

## Run the tests

```powershell
python -m pytest -q
```

Keep the `.env` file private and do not commit it to source control.

## Production deployment

The production container listens on `0.0.0.0:8001` inside Docker and joins the
external `tattoo_hysteria_net` network. It does not publish port 8001 to the
EC2 host. See [DEPLOYMENT.md](DEPLOYMENT.md) for the complete EC2, Compose,
CI/CD, secret, rollback, and backend-verification instructions.
