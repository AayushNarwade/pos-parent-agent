from flask import Flask, request, jsonify
import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
import pytz
from groq import Groq

# ----------------- Setup -----------------
app = Flask(__name__)
load_dotenv()

# Env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

XP_AGENT_URL = os.getenv("XP_AGENT_URL", "https://xp-agent.onrender.com/award_xp")
CALENDAR_AGENT_URL = os.getenv("CALENDAR_AGENT_URL", "https://calendar-agent.onrender.com/create_event")
EMAIL_AGENT_URL = os.getenv("EMAIL_AGENT_URL", "https://email-agent.onrender.com/create_draft")
RESEARCH_AGENT_URL = os.getenv("RESEARCH_AGENT_URL", "https://research-agent.onrender.com/research")
MESSAGING_AGENT_URL = os.getenv("MESSAGING_AGENT_URL", "https://messaging-agent.onrender.com/notify")

IST = pytz.timezone("Asia/Kolkata")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ----------------- System Prompt -----------------
SYSTEM_PROMPT = """
You are the reasoning core of the Present Operating System (POS).
Interpret user messages and output ONLY valid JSON for one of these intents:

TASK:
{
  "intent": "TASK",
  "task_name": "<title>",
  "result": "<expected outcome>",
  "purpose": "<why>",
  "massive_action_plan": ["<step1>", "<step2>"],
  "paei_role": "<Producer|Administrator|Entrepreneur|Integrator>",
  "due_date": "<ISO datetime or null>",
  "status": "To Do",
  "context": "<original user message>",
  "source": "Parent Agent"
}

CALENDAR:
{
  "intent": "CALENDAR",
  "title": "<event title>",
  "start_time": "<ISO start>",
  "end_time": "<ISO end>",
  "description": "<details>",
  "context": "<original user message>",
  "source": "Parent Agent"
}

EMAIL:
{
  "intent": "EMAIL",
  "to": "<recipient>",
  "context": "<email content>",
  "source": "Parent Agent"
}
"""

# ----------------- Helper: Generic HTTP Caller -----------------
def call_agent(url, payload, name="Agent", timeout=15):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:
        return 500, str(e)

# ----------------- Helper: Append Links to Notion -----------------
def append_links_to_notion(context, email_link=None, calendar_link=None):
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    # Find the page using the "Context" field
    query_url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    query_payload = {"filter": {"property": "Context", "rich_text": {"contains": context}}}

    response = requests.post(query_url, headers=headers, json=query_payload)
    if response.status_code != 200:
        print("❌ Notion query failed:", response.text)
        return

    results = response.json().get("results", [])
    if not results:
        print("⚠️ No matching task found for context:", context)
        return

    page_id = results[0]["id"]
    update_payload = {"properties": {}}

    if email_link:
        update_payload["properties"]["Email Link"] = {"url": email_link}
    if calendar_link:
        update_payload["properties"]["Calendar Link"] = {"url": calendar_link}

    update_url = f"https://api.notion.com/v1/pages/{page_id}"
    update_resp = requests.patch(update_url, headers=headers, json=update_payload)
    print("📎 Updated Notion links:", update_resp.status_code)

# ----------------- Routes -----------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "✅ Parent Agent Running with Notion Link Integration"}), 200

@app.route("/route", methods=["POST"])
def route_message():
    try:
        data = request.get_json(force=True)
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Missing 'message'"}), 400

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            temperature=0.25,
            max_tokens=600,
        )

        raw = completion.choices[0].message.content.strip()
        print("🧠 Groq Output:", raw)
        try:
            intent_data = json.loads(raw)
        except:
            return jsonify({"intent": "UNKNOWN", "raw": raw}), 200

        intent = intent_data.get("intent")

        # ---------------- INTENT HANDLING ----------------
        if intent == "CALENDAR":
            code, resp_text = call_agent(CALENDAR_AGENT_URL, intent_data, name="Calendar Agent")
            try:
                cal_response = json.loads(resp_text)
                calendar_link = cal_response.get("html_link")
                append_links_to_notion(context=message, calendar_link=calendar_link)
            except Exception as e:
                print("⚠️ Calendar link append failed:", e)
            return jsonify({"intent": "CALENDAR", "status": "Forwarded", "resp": resp_text}), 200

        elif intent == "EMAIL":
            code, resp_text = call_agent(EMAIL_AGENT_URL, intent_data, name="Email Agent")
            try:
                email_response = json.loads(resp_text)
                email_link = email_response.get("brevo_response", {}).get("messageId", "")
                append_links_to_notion(context=message, email_link=email_link)
            except Exception as e:
                print("⚠️ Email link append failed:", e)
            return jsonify({"intent": "EMAIL", "status": "Forwarded", "resp": resp_text}), 200

        else:
            return jsonify({"intent": "UNKNOWN", "raw": intent_data}), 200

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

# ----------------- MAIN -----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    print(f"🚀 Parent Agent running on port {port}")
    app.run(host="0.0.0.0", port=port)
