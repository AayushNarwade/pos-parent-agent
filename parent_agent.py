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

# ----------------- ENV -----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

XP_AGENT_URL = os.getenv("XP_AGENT_URL", "https://xp-agent.onrender.com/award_xp")
CALENDAR_AGENT_URL = os.getenv("CALENDAR_AGENT_URL", "https://calendar-agent-7ofo.onrender.com/create_event")
EMAIL_AGENT_URL = os.getenv("EMAIL_AGENT_URL", "https://email-agent.onrender.com/create_draft")
RESEARCH_AGENT_URL = os.getenv("RESEARCH_AGENT_URL", "https://research-agent.onrender.com/research")
MESSAGING_AGENT_URL = os.getenv("MESSAGING_AGENT_URL", "https://messaging-agent.onrender.com/notify")

IST = pytz.timezone("Asia/Kolkata")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ----------------- SYSTEM PROMPT -----------------
SYSTEM_PROMPT = """
You are the reasoning core of the Present Operating System (POS).
Interpret user input and return ONLY VALID JSON — no Markdown, no ```json``` formatting.

TASK:
{
  "intent": "TASK",
  "task_name": "<title>",
  "result": "<expected outcome>",
  "purpose": "<why>",
  "massive_action_plan": ["<step 1>", "<step 2>"],
  "paei_role": "<Producer|Administrator|Entrepreneur|Integrator>",
  "due_date": "<ISO datetime in Asia/Kolkata or null>",
  "status": "To Do",
  "context": "<original message>",
  "source": "Parent Agent"
}

COMPLETE_TASK:
{
  "intent": "COMPLETE_TASK",
  "task_name": "<title>",
  "status": "Completed",
  "context": "<message>",
  "source": "Parent Agent"
}

CALENDAR:
{
  "intent": "CALENDAR",
  "title": "<event title>",
  "start_time": "<ISO start>",
  "end_time": "<ISO end>",
  "description": "<event description>",
  "context": "<message>",
  "source": "Parent Agent"
}

EMAIL:
{
  "intent": "EMAIL",
  "to": "<recipient email>",
  "context": "<email purpose or brief content>",
  "source": "Parent Agent"
}

RESEARCH:
{
  "intent": "RESEARCH",
  "topic": "<topic to research>",
  "depth": "<summary|brief|detailed>",
  "context": "<message>",
  "source": "Parent Agent"
}

MESSAGE:
{
  "intent": "MESSAGE",
  "priority": "<P1|P2|P3>",
  "message": "<text to send>",
  "context": "<message>",
  "source": "Parent Agent"
}
"""

# ----------------- HELPERS -----------------
def clean_json_output(text: str) -> str:
    """Remove markdown formatting like ```json``` or triple backticks."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    text = text.replace("```", "").strip()
    return text


def call_agent(url, payload, name="Agent", timeout=30):
    """Call child agents with error safety."""
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:
        return 500, str(e)


def create_task_in_notion(task_data):
    """Create a Notion Task item and return Notion page ID."""
    now_iso = datetime.now(IST).isoformat()
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    # Properties
    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Task": {"title": [{"text": {"content": task_data.get("task_name", "Untitled Task")}}]},
            "Result (R)": {"rich_text": [{"text": {"content": task_data.get("result", "")}}]},
            "Purpose (P)": {"rich_text": [{"text": {"content": task_data.get("purpose", "")}}]},
            "Massive Action Plan (M)": {"rich_text": [{"text": {"content": ', '.join(task_data.get("massive_action_plan", []))}}]},
            "PAEI Role": {"select": {"name": task_data.get("paei_role", "Producer")}},
            "Status": {"select": {"name": task_data.get("status", "To Do")}},
            "Due Date": {"date": {"start": task_data.get("due_date")}} if task_data.get("due_date") else {"date": None},
            "XP": {"number": 0},
            "Created At": {"date": {"start": now_iso}},
            "Source": {"rich_text": [{"text": {"content": task_data.get("source", "Parent Agent")}}]},
            "Context": {"rich_text": [{"text": {"content": task_data.get("context", "")}}]},
        }
    }

    resp = requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload, timeout=15)
    if resp.status_code == 200:
        notion_id = resp.json().get("id")
        return 200, notion_id
    return resp.status_code, None


def update_notion_with_link(notion_id: str, field: str, link: str):
    """Append Calendar or Email link to Notion Task."""
    url = f"https://api.notion.com/v1/pages/{notion_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    payload = {
        "properties": {
            field: {"url": link}
        }
    }
    resp = requests.patch(url, headers=headers, json=payload, timeout=10)
    return resp.status_code


# ----------------- ROUTES -----------------
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

        if not client:
            return jsonify({"error": "Groq client not configured"}), 500

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
            temperature=0.2,
            max_tokens=600,
        )

        raw = completion.choices[0].message.content.strip()
        print("🧠 Raw Groq Output:", raw)
        cleaned_raw = clean_json_output(raw)

        try:
            intent_data = json.loads(cleaned_raw)
        except Exception as e:
            print("⚠️ JSON parsing failed:", e)
            return jsonify({"intent": "UNKNOWN", "raw": raw}), 200

        intent = intent_data.get("intent", "UNKNOWN").upper()

        # ------------- Intent Handling -------------
        if intent == "TASK":
            code, notion_id = create_task_in_notion(intent_data)
            return jsonify({"intent": "TASK", "notion_status": code, "notion_id": notion_id}), 200

        elif intent == "COMPLETE_TASK":
            code, resp_text = call_agent(XP_AGENT_URL, intent_data, "XP Agent")
            return jsonify({"intent": "COMPLETE_TASK", "xp_status": code, "xp_resp": resp_text}), 200

        elif intent == "CALENDAR":
            code, cal_resp = call_agent(CALENDAR_AGENT_URL, intent_data, "Calendar Agent")
            try:
                cal_data = json.loads(cal_resp)
                html_link = cal_data.get("html_link")
                # Optional: Append Calendar Link to Notion Task
                if "task_id" in intent_data and html_link:
                    update_notion_with_link(intent_data["task_id"], "Calendar Link", html_link)
            except Exception:
                pass
            return jsonify({"intent": "CALENDAR", "status": "Forwarded to Calendar Agent", "cal_status": code, "cal_resp": cal_resp}), 200

        elif intent == "EMAIL":
            code, email_resp = call_agent(EMAIL_AGENT_URL, intent_data, "Email Agent")
            try:
                email_data = json.loads(email_resp)
                email_link = email_data.get("brevo_response", {}).get("messageId", None)
                if "task_id" in intent_data and email_link:
                    update_notion_with_link(intent_data["task_id"], "Email Link", str(email_link))
            except Exception:
                pass
            return jsonify({"intent": "EMAIL", "status": "Forwarded to Email Agent", "email_status": code, "email_resp": email_resp}), 200

        elif intent == "RESEARCH":
            code, research_resp = call_agent(RESEARCH_AGENT_URL, intent_data, "Research Agent")
            return jsonify({"intent": "RESEARCH", "status": "Forwarded to Research Agent", "research_status": code, "research_resp": research_resp}), 200

        elif intent == "MESSAGE":
            code, msg_resp = call_agent(MESSAGING_AGENT_URL, intent_data, "Messaging Agent")
            return jsonify({"intent": "MESSAGE", "status": "Forwarded to Messaging Agent", "msg_status": code, "msg_resp": msg_resp}), 200

        else:
            return jsonify({"intent": "UNKNOWN", "raw": intent_data}), 200

    except Exception as e:
        print("❌ Parent Agent Error:", e)
        return jsonify({"error": str(e)}), 500


# ----------------- MAIN -----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    print(f"🚀 Parent Agent v2 running on port {port}")
    app.run(host="0.0.0.0", port=port)
