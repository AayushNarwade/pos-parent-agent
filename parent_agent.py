from flask import Flask, request, jsonify
import os
import json
import requests
from groq import Groq
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

# ========== ENVIRONMENT VARIABLES ==========
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")
XP_AGENT_URL = os.getenv("XP_AGENT_URL", "https://xp-agent.onrender.com/award_xp")
CALENDAR_AGENT_URL = os.getenv("CALENDAR_AGENT_URL", "http://127.0.0.1:10002/create_event")
EMAIL_AGENT_URL = os.getenv("EMAIL_AGENT_URL", "http://127.0.0.1:10003/create_draft")

print("🔧 Config:")
print("  GROQ_API_KEY:", bool(GROQ_API_KEY))
print("  NOTION_API_KEY:", bool(NOTION_API_KEY))
print("  NOTION_DATABASE_ID:", NOTION_DATABASE_ID)
print("  EMAIL_AGENT_URL:", EMAIL_AGENT_URL)

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
IST = pytz.timezone("Asia/Kolkata")

# ========== UPDATED SYSTEM PROMPT ==========
SYSTEM_PROMPT = """
You are the reasoning engine of the Present Operating System (POS).
You must classify and structure messages into actionable data.

If the message is a scheduling, calling, or reminder type (like 'remind me', 'schedule', 'call'):
Return ONLY valid JSON in this format:
{
  "intent": "TASK",
  "task_name": "<clean short task name, e.g., 'Call Client'>",
  "person_name": "<who the task involves, e.g., 'Aayush'>",
  "due_date": "<ISO8601 datetime in IST, if mentioned, else null>",
  "status": "To Do",
  "avatar": "Producer",
  "xp": 0
}

If the message indicates completion (like 'mark done', 'completed', 'finished'):
Return ONLY valid JSON in this format:
{
  "intent": "COMPLETE_TASK",
  "task_name": "<the task name or key phrase>",
  "avatar": "Producer",
  "status": "Completed"
}

If the message is an email-type instruction (like 'send an email', 'thank John', 'write to Sarah'):
Return ONLY valid JSON in this format:
{
  "intent": "EMAIL",
  "to": "<recipient email if known, else null>",
  "context": "<the content or purpose of the email>"
}

If it's a question or non-task, return:
{
  "intent": "RESEARCH",
  "question": "<the user's query>"
}

Do NOT include explanations or markdown — only JSON.
"""

# ========== HELPERS ==========

def find_task_in_notion(task_name):
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    body = {
        "filter": {"property": "Task", "title": {"contains": task_name}}
    }
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    r = requests.post(url, headers=headers, json=body)
    if r.status_code != 200:
        print("❌ Error searching task:", r.text)
        return None
    results = r.json().get("results", [])
    if not results:
        print("⚠️ Task not found in Notion.")
        return None
    return results[0]["id"]


def update_task_status(page_id, new_status="Completed"):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    body = {"properties": {"Status": {"select": {"name": new_status}}}}
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    r = requests.patch(url, headers=headers, json=body)
    if r.status_code not in [200, 201]:
        print("❌ Failed to update task status:", r.text)
        return False
    print(f"✅ Task status updated to {new_status}")
    return True


def call_xp_agent(task_name, avatar="Producer", reason="Task Completed", due_date=None):
    payload = {
        "task_name": task_name,
        "avatar": avatar,
        "due_date": due_date,
        "reason": reason,
    }
    try:
        r = requests.post(XP_AGENT_URL, json=payload)
        print(f"📩 XP Agent Response: {r.status_code} - {r.text}")
        return r.status_code, r.text
    except Exception as e:
        print("❌ Failed to contact XP Agent:", e)
        return 500, str(e)


def call_calendar_agent(title, description, start_iso, end_iso=None):
    payload = {
        "title": title,
        "description": description or "",
        "start_time": start_iso,
        "end_time": end_iso or (datetime.fromisoformat(start_iso) + timedelta(minutes=30)).isoformat(),
    }

    try:
        r = requests.post(CALENDAR_AGENT_URL, json=payload, timeout=6)
        print("📆 Calendar Agent Response:", r.status_code, r.text)
        return r.status_code, r.text
    except Exception as e:
        print("❌ Failed to reach Calendar Agent:", e)
        return 500, str(e)


def call_email_agent(to_email, context):
    """
    Calls the AI Email Agent to create an email draft.
    Render free instances can have cold-start delays, so use a longer timeout.
    """
    payload = {"to": to_email, "context": context}
    try:
        r = requests.post(EMAIL_AGENT_URL, json=payload, timeout=35)  # increased from 10 → 35 sec
        print("📧 Email Agent Response:", r.status_code, r.text)
        return r.status_code, r.text
    except requests.exceptions.ReadTimeout:
        print("⚠️ Email Agent took too long to respond (timeout). Consider waking it manually.")
        return 504, "Email Agent timeout — service may be waking up."
    except Exception as e:
        print("❌ Failed to contact Email Agent:", e)
        return 500, str(e)



# ========== ENDPOINT ==========
@app.route("/route", methods=["POST"])
def route_message():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400

        # Inject current IST time
        current_time_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z")
        system_prompt_with_date = (
            SYSTEM_PROMPT
            + f"\n\nToday's date and time (IST): {current_time_ist}."
            + " Always generate due_date relative to this current date."
        )

        if not client:
            return jsonify({"error": "Groq client not configured (missing GROQ_API_KEY)"}), 500

        # --- Groq classification ---
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt_with_date},
                {"role": "user", "content": message},
            ],
            temperature=0.3,
            max_tokens=300,
        )

        raw_reply = completion.choices[0].message.content.strip()
        print("🧠 Raw Groq Response:", raw_reply)

        # Parse JSON
        try:
            result = json.loads(raw_reply)
        except Exception:
            # Return raw reply to help debugging / fallback
            return jsonify({"intent": "UNKNOWN", "raw_reply": raw_reply}), 200

        intent = result.get("intent")

        # ------------------ TASK CREATION ------------------
        if intent == "TASK":
            task_name = result.get("task_name", "Untitled Task")
            person_name = result.get("person_name", "Unknown")
            due_date = result.get("due_date")
            status = result.get("status", "To Do")
            avatar = result.get("avatar", "Producer")
            xp = result.get("xp", 0)

            now_ist = datetime.now(IST)
            if not due_date or str(now_ist.year) not in due_date:
                due_date = now_ist.isoformat()

            payload = {
                "parent": {"database_id": NOTION_DATABASE_ID},
                "properties": {
                    "Task": {"title": [{"text": {"content": task_name}}]},
                    "Name": {"rich_text": [{"text": {"content": person_name}}]},
                    "Status": {"select": {"name": status}},
                    "Avatar": {"select": {"name": avatar}},
                    "XP": {"number": xp},
                    "Due Date": {"date": {"start": due_date}},
                },
            }

            headers = {
                "Authorization": f"Bearer {NOTION_API_KEY}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json",
            }

            notion_resp = requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload)
            print("📝 Notion Response:", notion_resp.status_code, notion_resp.text)

            if notion_resp.status_code not in (200, 201):
                return jsonify({"error": "Failed to add to Notion", "details": notion_resp.text}), 500

            # Optional calendar
            try:
                dt_start = datetime.fromisoformat(due_date)
                dt_end = dt_start + timedelta(minutes=30)
                call_calendar_agent(task_name, f"Auto-scheduled task: {task_name}", dt_start.isoformat(), dt_end.isoformat())
            except Exception as e:
                print("⚠️ Calendar scheduling skipped:", e)

            return jsonify({
                "intent": "TASK",
                "status": "Task added to Notion (and Calendar if due_date found)",
                "task_name": task_name,
                "person_name": person_name,
                "xp": xp,
                "due_date": due_date,
            }), 200

        # ------------------ TASK COMPLETION ------------------
        elif intent == "COMPLETE_TASK":
            task_name = result.get("task_name", "").strip()
            avatar = result.get("avatar", "Producer")

            page_id = find_task_in_notion(task_name)
            if not page_id:
                return jsonify({"status": "error", "message": f"Task '{task_name}' not found"}), 404

            updated = update_task_status(page_id, "Completed")
            if not updated:
                return jsonify({"status": "error", "message": "Failed to update Notion task"}), 500

            xp_status, xp_response = call_xp_agent(task_name, avatar, "Task Completed")

            return jsonify({
                "intent": "COMPLETE_TASK",
                "status": "Task marked as completed ✅",
                "task_name": task_name,
                "xp_agent_response": xp_response,
            }), 200

        # ------------------ EMAIL HANDLER ------------------
        elif intent == "EMAIL":
            # If the model didn't provide a to/context, use safe fallbacks:
            to_email = result.get("to") or "john@example.com"
            # Prefer explicit 'context' from model; otherwise fallback to the original message
            context = result.get("context") or message
            if not context.strip():
                context = message

            payload_preview = {"to": to_email, "context": context}
            print("📨 Email Payload to Agent:", payload_preview)

            status, resp = call_email_agent(to_email, context)
            if status not in [200, 201]:
                print("❌ Email Agent Failure:", resp)
                return jsonify({
                    "error": "Email agent failed",
                    "sent_payload": payload_preview,
                    "details": resp
                }), 500

            # parse response from email agent if possible
            try:
                email_resp = json.loads(resp) if isinstance(resp, str) else resp
            except Exception:
                email_resp = {"raw_response": resp}

            return jsonify({
                "intent": "EMAIL",
                "status": "Email draft created successfully ✅",
                "response": email_resp
            }), 200

        # ------------------ RESEARCH HANDLER ------------------
        elif intent == "RESEARCH":
            return jsonify({
                "intent": "RESEARCH",
                "response": "Research processing coming soon!"
            }), 200

        else:
            return jsonify({"intent": "UNKNOWN", "response": raw_reply}), 200

    except Exception as e:
        print("❌ Server Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Parent Agent running ✅"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Render injects its own $PORT
    app.run(host="0.0.0.0", port=port)

