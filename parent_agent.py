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
EMAIL_AGENT_URL = os.getenv("EMAIL_AGENT_URL", "https://email-agent-x7n0.onrender.com/create_draft")  # updated
RESEARCH_AGENT_URL = os.getenv("RESEARCH_AGENT_URL", "https://research-agent.onrender.com/research")
MESSAGING_AGENT_URL = os.getenv("MESSAGING_AGENT_URL", "https://messaging-agent.onrender.com/notify")

IST = pytz.timezone("Asia/Kolkata")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ----------------- SYSTEM PROMPT -----------------
SYSTEM_PROMPT = """
You are the reasoning engine of the Present Operating System (POS).
Interpret user input and return ONLY VALID JSON (no markdown formatting).

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
  "context": "<email purpose or brief content>",
  "source": "Parent Agent"
}
"""

# ----------------- HELPERS -----------------
def clean_json_output(text: str) -> str:
    """Remove markdown formatting like ```json``` or triple backticks."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").replace("json", "", 1).strip()
    return text.replace("```", "").strip()


def call_agent(url, payload, name="Agent", timeout=25):
    """Send data to a child agent with timeout handling."""
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r.status_code, r.text
    except Exception as e:
        print(f"⚠️ {name} connection error:", e)
        return 500, str(e)


def create_task_in_notion(task_data):
    """Create a task in Notion with all required fields."""
    now_iso = datetime.now(IST).isoformat()
    due_date = task_data.get("due_date")

    payload = {
        "parent": {"database_id": NOTION_DATABASE_ID},
        "properties": {
            "Task": {"title": [{"text": {"content": task_data.get("task_name", "Untitled Task")}}]},
            "Result (R)": {"rich_text": [{"text": {"content": task_data.get("result", "")}}]},
            "Purpose (P)": {"rich_text": [{"text": {"content": task_data.get("purpose", "")}}]},
            "Massive Action Plan (M)": {"rich_text": [{"text": {"content": ', '.join(task_data.get("massive_action_plan", []))}}]},
            "PAEI Role": {"select": {"name": task_data.get("paei_role", "Producer")}},
            "Status": {"select": {"name": task_data.get("status", "To Do")}},
            "Due Date": {"date": {"start": due_date}} if due_date else {"date": None},
            "Created At": {"date": {"start": now_iso}},
            "Source": {"rich_text": [{"text": {"content": task_data.get("source", "Parent Agent")}}]},
            "Context": {"rich_text": [{"text": {"content": task_data.get("context", "")}}]},
        }
    }

    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    resp = requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload, timeout=15)
    if resp.status_code in [200, 201]:
        notion_id = resp.json().get("id")
        print(f"✅ Task created in Notion: {notion_id}")
        return 200, notion_id
    print("❌ Notion creation failed:", resp.text)
    return resp.status_code, None


def update_notion_with_link(notion_id: str, field: str, link: str):
    """Append Calendar or Email link to Notion Task."""
    url = f"https://api.notion.com/v1/pages/{notion_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    payload = {"properties": {field: {"url": link}}}
    resp = requests.patch(url, headers=headers, json=payload, timeout=10)
    if resp.status_code in [200, 201]:
        print(f"🔗 Updated {field} link in Notion successfully.")
    else:
        print(f"⚠️ Failed to update {field}: {resp.text}")
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

        # LLM determines intent
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
        cleaned_raw = clean_json_output(raw)
        print("🧠 Parsed:", cleaned_raw)

        try:
            intent_data = json.loads(cleaned_raw)
        except Exception as e:
            return jsonify({"intent": "UNKNOWN", "raw": raw, "error": str(e)}), 200

        intent = intent_data.get("intent", "UNKNOWN").upper()

        # ---------- CALENDAR ----------
        if intent == "CALENDAR":
            task_info = {
                "task_name": intent_data.get("title", "Untitled Event"),
                "result": "Calendar event scheduled",
                "purpose": "Manage meetings and events",
                "massive_action_plan": ["Schedule meeting", "Confirm details"],
                "paei_role": "Administrator",
                "status": "To Do",
                "context": intent_data.get("context", ""),
                "source": "Parent Agent"
            }
            notion_status, notion_id = create_task_in_notion(task_info)

            code, cal_resp = call_agent(CALENDAR_AGENT_URL, intent_data, "Calendar Agent")
            try:
                cal_data = json.loads(cal_resp)
                html_link = cal_data.get("html_link") or cal_data.get("calendar_link")
                if html_link and notion_id:
                    update_notion_with_link(notion_id, "Calendar Link", html_link)
            except Exception as e:
                print("⚠️ Calendar link update failed:", e)

            return jsonify({
                "intent": "CALENDAR",
                "status": "Forwarded to Calendar Agent",
                "notion_id": notion_id,
                "notion_status": notion_status,
                "cal_status": code,
                "cal_resp": cal_resp
            }), 200

        # ---------- EMAIL ----------
        elif intent == "EMAIL":
            task_info = {
                "task_name": "Email Communication",
                "result": "Email draft sent",
                "purpose": "Handle communication tasks",
                "massive_action_plan": ["Draft email", "Send message"],
                "paei_role": "Administrator",
                "status": "To Do",
                "context": intent_data.get("context", ""),
                "source": "Parent Agent"
            }
            notion_status, notion_id = create_task_in_notion(task_info)

            # Add static email
            intent_data["to"] = "narwadeaayush169@gmail.com"

            code, email_resp = call_agent(EMAIL_AGENT_URL, intent_data, "Email Agent")
            try:
                email_data = json.loads(email_resp)
                message_id = email_data.get("brevo_response", {}).get("messageId", "")
                if message_id and notion_id:
                    email_url = f"https://app.brevo.com/email/{message_id}"
                    update_notion_with_link(notion_id, "Email Link", email_url)
            except Exception as e:
                print("⚠️ Email link update failed:", e)

            return jsonify({
                "intent": "EMAIL",
                "status": "Forwarded to Email Agent",
                "notion_id": notion_id,
                "notion_status": notion_status,
                "email_status": code,
                "email_resp": email_resp
            }), 200

        # ---------- UNKNOWN ----------
        else:
            return jsonify({"intent": "UNKNOWN", "raw": intent_data}), 200

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


# ----------------- MAIN -----------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    print(f"🚀 Parent Agent v2 running on port {port}")
    app.run(host="0.0.0.0", port=port)
