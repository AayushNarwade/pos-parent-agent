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

client = Groq(api_key=GROQ_API_KEY)
IST = pytz.timezone("Asia/Kolkata")

# ========== SYSTEM PROMPT ==========
SYSTEM_PROMPT = """
You are the reasoning engine of the Present Operating System (POS).
You must classify and structure messages into actionable data.

If the message is a task (like 'remind me', 'schedule', 'call', 'email'):
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

If it's a question or non-task, return:
{
  "intent": "RESEARCH",
  "question": "<the user's query>"
}

Do NOT include explanations or markdown — only JSON.
"""

# ========== HELPERS ==========

def find_task_in_notion(task_name):
    """
    Search the Notion Task DB for a given task name.
    Returns the page_id if found.
    """
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    body = {
        "filter": {
            "property": "Task",
            "title": {"contains": task_name}
        }
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
    """
    Updates task status in Notion DB.
    """
    url = f"https://api.notion.com/v1/pages/{page_id}"
    body = {
        "properties": {
            "Status": {"select": {"name": new_status}}
        }
    }
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
    """
    Sends task completion data to XP Agent to log XP.
    """
    payload = {
        "task_name": task_name,
        "avatar": avatar,
        "due_date": due_date,
        "reason": reason
    }
    try:
        r = requests.post(XP_AGENT_URL, json=payload)
        print(f"📩 XP Agent Response: {r.status_code} - {r.text}")
        return r.status_code, r.text
    except Exception as e:
        print("❌ Failed to contact XP Agent:", e)
        return 500, str(e)

def call_calendar_agent(title, description, start_iso, end_iso=None):
    """
    Sends event details to the Calendar Agent for scheduling in Google Calendar.
    """
    payload = {
        "title": title,
        "description": description or "",
        "start_time": start_iso,
        "end_time": end_iso or (datetime.fromisoformat(start_iso) + timedelta(minutes=30)).isoformat()
    }

    try:
        r = requests.post(CALENDAR_AGENT_URL, json=payload, timeout=6)
        print("📆 Calendar Agent Response:", r.status_code, r.text)
        return r.status_code, r.text
    except Exception as e:
        print("❌ Failed to reach Calendar Agent:", e)
        return 500, str(e)


# ========== ENDPOINT ==========
@app.route("/route", methods=["POST"])
def route_message():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400

        # --- Inject current IST date/time for context ---
        current_time_ist = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z")
        system_prompt_with_date = (
            SYSTEM_PROMPT
            + f"\n\nToday's date and time (IST): {current_time_ist}."
            + " Always generate due_date relative to this current date."
        )

        # --- Call Groq ---
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

        # --- Parse JSON safely ---
        try:
            result = json.loads(raw_reply)
        except Exception:
            return jsonify({"intent": "UNKNOWN", "raw_reply": raw_reply}), 200

        # ------------------  HANDLE TASK CREATION  ------------------
        if result.get("intent") == "TASK":
            task_name = result.get("task_name", "Untitled Task")
            person_name = result.get("person_name", "Unknown")
            due_date = result.get("due_date")
            status = result.get("status", "To Do")
            avatar = result.get("avatar", "Producer")
            xp = result.get("xp", 0)

            now_ist = datetime.now(IST)
            if not due_date or not str(now_ist.year) in due_date:
                due_date = now_ist.isoformat()

            # --- Build Notion Payload ---
            safe_person_name = result.get("person_name", "General") or "General"

            payload = {
                "parent": {"database_id": NOTION_DATABASE_ID},
                "properties": {
                    "Task": {"title": [{"text": {"content": task_name or 'Untitled Task'}}]},
                    "Name": {"rich_text": [{"text": {"content": safe_person_name}}]},
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

            notion_resp = requests.post(
                "https://api.notion.com/v1/pages", headers=headers, json=payload
            )

            print("📝 Notion Response:", notion_resp.status_code, notion_resp.text)

            if notion_resp.status_code not in (200, 201):
                return jsonify({"error": "Failed to add to Notion", "details": notion_resp.text}), 500

            # ---------- Optional: Schedule to Calendar if due_date is valid ----------
            if due_date:
                try:
                    dt_start = datetime.fromisoformat(due_date)
                    dt_end = dt_start + timedelta(minutes=30)
                    start_iso = dt_start.isoformat()
                    end_iso = dt_end.isoformat()

                    cal_status, cal_resp = call_calendar_agent(
                        title=f"{task_name}",
                        description=f"Auto-scheduled task: {task_name}",
                        start_iso=start_iso,
                        end_iso=end_iso
                    )
                    print("📅 Calendar Integration:", cal_status, cal_resp)
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

        # ------------------  HANDLE TASK COMPLETION  ------------------
        elif result.get("intent") == "COMPLETE_TASK":
            task_name = result.get("task_name", "").strip()
            avatar = result.get("avatar", "Producer")

            page_id = find_task_in_notion(task_name)
            if not page_id:
                return jsonify({"status": "error", "message": f"Task '{task_name}' not found"}), 404

            updated = update_task_status(page_id, "Completed")
            if not updated:
                return jsonify({"status": "error", "message": "Failed to update Notion task"}), 500

            # Call XP Agent
            xp_status, xp_response = call_xp_agent(task_name, avatar, "Task Completed")

            return jsonify({
                "intent": "COMPLETE_TASK",
                "status": "Task marked as completed ✅",
                "task_name": task_name,
                "xp_agent_response": xp_response,
            }), 200

        # ------------------  HANDLE RESEARCH  ------------------
        elif result.get("intent") == "RESEARCH":
            return jsonify({
                "intent": "RESEARCH",
                "response": "Research processing coming soon!",
            }), 200

        # ------------------  UNKNOWN INTENT  ------------------
        else:
            return jsonify({"intent": "UNKNOWN", "response": raw_reply}), 200

    except Exception as e:
        print("❌ Server Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "Parent Agent running ✅"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
