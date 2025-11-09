from flask import Flask, request, jsonify
import os, json, re
from dotenv import load_dotenv
from groq import Groq
from flask_cors import CORS

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.3-70b-specdevi"  # stable, JSON-friendly

app = Flask(__name__)
CORS(app)

SYSTEM_PROMPT = """
You are the 'Parent Agent' for the Present Operating System (POS).
TIMEZONE: Asia/Kolkata (IST, UTC+05:30).

You must classify the user message into exactly one intent:
- TASK      → actions, reminders, meetings, habits, follow-ups
- RESEARCH  → questions, info lookup, factual queries
- UNKNOWN   → chit chat, unclear, off-topic

STRICT OUTPUT: JSON only. Never add text outside JSON.

For TASK intent, extract a structured task:
- title: short, capitalized action (strip fillers like "remind me to", "please")
- datetime_iso: an ISO8601 datetime in IST with offset, if a time/date is implied
  * AUTO_DATE = YES: If time present but date missing, choose today if time is in the future,
    otherwise tomorrow. If date words like "tomorrow", "Monday", or "on 12th" exist, resolve to that date.
  * Interpret ambiguous “at 10” as 10:00 **AM** unless explicitly PM.
  * Preserve +05:30 offset, e.g. "2025-11-10T16:00:00+05:30".

If no time is implied, set datetime_iso to null.

For RESEARCH and UNKNOWN, just echo cleaned text in "data".

Return one of these shapes:

1) TASK:
{
  "intent": "TASK",
  "data": "<cleaned user text>",
  "task": {
    "title": "<clean, short, capitalized>",
    "datetime_iso": "<ISO with +05:30 or null>"
  }
}

2) RESEARCH:
{
  "intent": "RESEARCH",
  "data": "<question>"
}

3) UNKNOWN:
{
  "intent": "UNKNOWN",
  "data": "<cleaned text>"
}
"""

def extract_json(text: str):
    """Be tolerant if the model wraps JSON in extra text."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise

@app.route("/route", methods=["POST"])
def route_message():
    try:
        user_input = (request.get_json() or {}).get("message", "").strip()
        if not user_input:
            return jsonify({"intent": "UNKNOWN", "data": ""}), 200

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        result = extract_json(raw)

        # Hard guarantees
        if result.get("intent") == "TASK":
            task = result.get("task") or {}
            task.setdefault("title", user_input)
            task.setdefault("datetime_iso", None)
            result["task"] = task
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
