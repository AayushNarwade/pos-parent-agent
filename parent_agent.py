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

# ----------------- System prompt (expects context + source) -----------------
SYSTEM_PROMPT = """
You are the core reasoning engine of the Present Operating System (POS).
Interpret user input and return ONLY valid JSON according to the cases below.

TASK (create a task) -> include RPM + PAEI + trace fields:
{
  "intent": "TASK",
  "task_name": "<short clear title>",
  "result": "<expected outcome>",
  "purpose": "<why the user wants this>",
  "massive_action_plan": ["<step 1>", "<step 2>"],
  "paei_role": "<Producer|Administrator|Entrepreneur|Integrator>",
  "due_date": "<ISO datetime in Asia/Kolkata or null>",
  "status": "To Do",
  "context": "<original user message>",
  "source": "Parent Agent"
}

COMPLETE_TASK:
{
  "intent": "COMPLETE_TASK",
  "task_name": "<task title>",
  "status": "Completed",
  "context": "<original user message>",
  "source": "Parent Agent"
}

CALENDAR (schedule event):
{
  "intent": "CALENDAR",
  "title": "<event title>",
  "start_time": "<ISO start>",
  "end_time": "<ISO end>",
  "description": "<event description>",
  "context": "<original user message>",
  "source": "Parent Agent"
}

EMAIL:
{
  "intent": "EMAIL",
  "to": "<recipient email or null>",
  "context": "<email purpose or brief content>",
  "source": "Parent Agent"
}

RESEARCH:
{
  "intent": "RESEARCH",
  "topic": "<what to research>",
  "depth": "<summary|brief|detailed>",
  "context": "<original user message>",
  "source": "Parent Agent"
}

MESSAGE:
{
  "intent": "MESSAGE",
  "priority": "<P1|P2|P3>",
  "message": "<text to send>",
  "context": "<original user message>",
  "source": "Parent Agent"
}
"""

# ----------------- Helpers -----------------
def call_agent(url, payload, name="Agent", timeout=15):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:
        return 500, str(e)

def create_task_in_notion(task_data):
    """
    Insert a task into Notion using the final schema including Source and Context.
    """
    now_iso = datetime.now(IST).isoformat()
    # safe getters and fallbacks
    task_name = task_data.get("task_name", "Untitled Task")
    result = task_data.get("result", "")
    purpose = task_data.get("purpose", "")
    map_list = task_data.get("massive_action_plan") or []
    paei_role = task_data.get("paei_role", "Producer")
    status = task_data.get("status", "To Do")
    due_date = task_data.get("due_date")
    context_txt = task_data.get("context", "")
    source_txt = task_data.get("source", "Parent Agent")

    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Task": {"title": [{"text": {"content": task_name}}]},
            "Result (R)": {"rich_text": [{"text": {"content": result}}]},
            "Purpose (P)": {"rich_text": [{"text": {"content": purpose}}]},
            "Massive Action Plan (M)": {"rich_text": [{"text": {"content": ', '.join(map_list)}}]},
            "PAEI Role": {"select": {"name": paei_role}},
            "Status": {"select": {"name": status}},
            "Due Date": {"date": {"start": due_date}} if due_date else {"date": None},
            "XP": {"number": 0},
            "Created At": {"date": {"start": now_iso}},
            "Source": {"rich_text": [{"text": {"content": source_txt}}]},
            "Context": {"rich_text": [{"text": {"content": context_txt}}]}
        }
    }

    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    resp = requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload, timeout=15)
    return resp.status_code, resp.text

# ----------------- Routes -----------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "✅ POS Parent Agent v2 Running"}), 200

@app.route("/route", methods=["POST"])
def route_message():
    try:
        data = request.get_json(force=True)
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Missing 'message'"}), 400

        # include original message in system/user context (helps Groq produce 'context')
        if not client:
            return jsonify({"error": "Groq client not configured (missing GROQ_API_KEY)"}), 500

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
        print("🧠 Groq output:", raw)

        try:
            intent_data = json.loads(raw)
        except Exception as e:
            # return raw for debug if model didn't produce valid JSON
            return jsonify({"intent": "UNKNOWN", "raw": raw}), 200

        intent = intent_data.get("intent")

        # handle intents
        if intent == "TASK":
            # ensure context + source exist
            if "context" not in intent_data:
                intent_data["context"] = message
            if "source" not in intent_data:
                intent_data["source"] = "Parent Agent"

            code, resp_text = create_task_in_notion(intent_data)
            return jsonify({"intent": "TASK", "status": "Task created", "notion_status": code, "notion_response": resp_text}), 200

        elif intent == "COMPLETE_TASK":
            # forward to XP Agent (XP Agent will update notion / ledger)
            if "context" not in intent_data:
                intent_data["context"] = message
            if "source" not in intent_data:
                intent_data["source"] = "Parent Agent"

            code, resp_text = call_agent(XP_AGENT_URL, intent_data, name="XP Agent")
            return jsonify({"intent": "COMPLETE_TASK", "status": "Forwarded to XP Agent", "xp_status": code, "xp_resp": resp_text}), 200

        elif intent == "EMAIL":
            if "context" not in intent_data:
                intent_data["context"] = message
            if "source" not in intent_data:
                intent_data["source"] = "Parent Agent"

            code, resp_text = call_agent(EMAIL_AGENT_URL, intent_data, name="Email Agent")
            return jsonify({"intent": "EMAIL", "status": "Forwarded to Email Agent", "email_status": code, "email_resp": resp_text}), 200

        elif intent == "CALENDAR":
            if "context" not in intent_data:
                intent_data["context"] = message
            if "source" not in intent_data:
                intent_data["source"] = "Parent Agent"

            code, resp_text = call_agent(CALENDAR_AGENT_URL, intent_data, name="Calendar Agent")
            return jsonify({"intent": "CALENDAR", "status": "Forwarded to Calendar Agent", "cal_status": code, "cal_resp": resp_text}), 200

        elif intent == "RESEARCH":
            if "context" not in intent_data:
                intent_data["context"] = message
            if "source" not in intent_data:
                intent_data["source"] = "Parent Agent"

            code, resp_text = call_agent(RESEARCH_AGENT_URL, intent_data, name="Research Agent")
            return jsonify({"intent": "RESEARCH", "status": "Forwarded to Research Agent", "research_status": code, "research_resp": resp_text}), 200

        elif intent == "MESSAGE":
            if "context" not in intent_data:
                intent_data["context"] = message
            if "source" not in intent_data:
                intent_data["source"] = "Parent Agent"

            code, resp_text = call_agent(MESSAGING_AGENT_URL, intent_data, name="Messaging Agent")
            return jsonify({"intent": "MESSAGE", "status": "Forwarded to Messaging Agent", "msg_status": code, "msg_resp": resp_text}), 200

        else:
            return jsonify({"intent": "UNKNOWN", "raw": intent_data}), 200

    except Exception as e:
        print("❌ Parent error:", e)
        return jsonify({"error": str(e)}), 500

# ----------------- Main -----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
