from flask import Flask, request, jsonify
import os
import json
from groq import Groq
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)

# ✅ Load environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ==============================
# SYSTEM PROMPT
# ==============================
SYSTEM_PROMPT = """
You are the core reasoning engine of the Present Operating System (POS).
Your job is to classify and structure user messages into either TASK or RESEARCH.

If the user’s message is an actionable item (“remind”, “schedule”, “email”, “call”), 
return:
{
  "intent": "TASK",
  "data": "<original_message>",
  "task": {
    "title": "<clean concise task title>",
    "datetime_iso": "<ISO8601 datetime in IST, or null if not specified>"
  }
}

If it’s a question, return:
{
  "intent": "RESEARCH",
  "data": "<question>"
}

Always produce valid JSON — no markdown, no code blocks, no explanations.
If unsure, still produce a JSON with "intent": "UNKNOWN".
"""

# ==============================
# POST ENDPOINT
# ==============================
@app.route("/route", methods=["POST"])
def route_message():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"error": "Empty message"}), 400

        # Call Groq for reasoning
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            temperature=0.3,
            max_tokens=300
        )

        raw_reply = completion.choices[0].message.content.strip()

        # Attempt to parse JSON safely
        try:
            parsed = json.loads(raw_reply)
        except Exception:
            parsed = {"intent": "UNKNOWN", "data": message, "raw": raw_reply}

        print("🔍 Parsed JSON:", parsed)
        return jsonify(parsed)

    except Exception as e:
        print("❌ Server Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
