"""Replay a client conversation through the AI brain from the terminal.

Usage (from the project root):
    .venv\\Scripts\\python.exe scripts\\replay_chat.py                 # built-in scalp chat
    .venv\\Scripts\\python.exe scripts\\replay_chat.py --interactive   # type messages live
    .venv\\Scripts\\python.exe scripts\\replay_chat.py --file chat.json
    .venv\\Scripts\\python.exe scripts\\replay_chat.py --source outlook --image https://...

A --file is a JSON list of client messages (strings). Each turn feeds the
bot's own previous replies back as history, exactly like production.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_brain.processor import StudioAIBrain  # noqa: E402
from ai_brain.schemas import Message  # noqa: E402

SCALP_CHAT = [
    "hello, I want to make a tattoo.\nHow much do you guys usually charge?",
    "The placement is on my head; I'm basically bald.\nSo I'm so embarrassed "
    "about that, and I'm gonna tattoo my scalp. I have a reference image, though.",
    "I don't really know about the size.\nIt has to cover my whole head, "
    "possibly the whole hair area",
    "I mean, yeah, from front to back and right to left, both maybe 30cm around.",
    "I don't know about the artists. Could you please help me to choose a "
    "suitable artist for that?",
    "studio visit",
    "today 5 pm",
    "sandra",
    "new tattoo",
]

DETAIL_FIELDS = (
    "tattoo_idea",
    "placement",
    "size_estimate_cm",
    "color_preference",
    "style_tags",
    "preferred_artist",
    "artist_preference_mode",
    "appointment_type",
    "date",
    "time",
    "tattoo_project_type",
    "suggested_artist",
)


def _print_turn(index: int, message: str, result, verbose: bool) -> None:
    print(f"\n{'=' * 78}\n[{index}] CLIENT: {message}")
    print(
        f"[{index}] BOT ({len(result.draft_reply)} chars): {result.draft_reply}"
    )
    print(
        f"    risk={result.risk_level}  auto_reply={result.auto_reply}  "
        f"staff_review={result.staff_review_required}"
    )
    print(f"    missing={list(result.missing_information)}")
    if verbose:
        for field in DETAIL_FIELDS:
            print(f"    {field}={getattr(result, field)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", help="JSON list of client messages")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--source",
        default="whatsapp",
        choices=("whatsapp", "outlook", "vcita"),
    )
    parser.add_argument("--name", default="Fahim Sarker", help="lead name")
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="reference image URL attached to the first message (repeatable)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="hide extracted fields, show only replies",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    # Surfaces "Draft reply LLM fallback used: <reason>" when the model's
    # draft is rejected and the template reply is sent instead.
    logging.basicConfig(level=logging.WARNING, format="    !! %(message)s")

    brain = StudioAIBrain()
    state = {
        "lead": {"name": args.name, "source": args.source},
        "intake": {"source": args.source},
    }
    history: list[Message] = []

    if args.interactive:
        messages = None
        print("Type client messages. Empty line or Ctrl+C to quit.")
    elif args.file:
        messages = json.loads(Path(args.file).read_text(encoding="utf-8"))
    else:
        messages = SCALP_CHAT

    index = 0
    while True:
        if messages is None:
            try:
                message = input("\nCLIENT> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not message:
                break
        else:
            if index >= len(messages):
                break
            message = messages[index]

        images = args.image if index == 0 else []
        history.append(Message(role="user", content=message))
        result = brain.process_inquiry(
            current_message=message,
            new_image_urls=images,
            existing_db_state=state,
            recent_chat_history=list(history),
            message_source=args.source,
        )
        index += 1
        _print_turn(index, message, result, verbose=not args.quiet)
        history.append(Message(role="assistant", content=result.draft_reply))


if __name__ == "__main__":
    main()
