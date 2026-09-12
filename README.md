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
  "message_source": "whatsapp",
  "new_image_urls": [],
  "existing_db_state": {},
  "recent_chat_history": [
    {"role": "user", "content": "I am thinking about a floral design."}
  ]
}
```

Set `message_source` to `outlook` to receive a professional email draft that
requests all missing intake details in one message. When omitted, it defaults
to the `source` stored in `existing_db_state.intake` or
`existing_db_state.lead`, then falls back to `whatsapp`. WhatsApp keeps the
concise conversational reply format. Outlook and Gmail quoted reply threads
are removed before extraction so previous studio questions cannot be mistaken
for new client answers.

The structured `appointment_type` response field is either `online`,
`studio_visit`, or an empty string while it is still unknown. Client-facing
replies display these choices naturally as “online” and “studio visit”.

Use `/api/v1/inquiries/telegram-summary` for high-risk inquiries. It returns a
staff summary together with the generated draft reply. A summary is returned
only when the latest analysis is high risk and has no missing intake items.
Telegram consumers should send the returned `telegram_message` for the same
inquiry and must not merge prices, schedules, service codes, or draft replies
from older intake records. Request and active-intake association remains the
calling backend's responsibility.

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
