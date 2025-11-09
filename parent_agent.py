from flask import Flask, request, jsonify
import os
import json
import requests
from groq import Groq
from datetime import datetime
import pytz

app = Flask(__name__)

# ========== ENVIRONMENT VARIABLES ==========
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID")

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
  "task_name": "<clean, short task name>",
  "due_date": "<ISO8601 datetime in IST, if mentioned, else null>",
  "status": "To Do",
  "avatar": "Producer"
}

If it's a question or non-task, return:
{
  "intent": "RESEARCH",
  "question": "<the user's query>"
}

Do NOT include explanations or markdown — only JSON.
"""

# ========== ENDPOINTS ==========
@app.route("/route", methods=["POST"])
def route_message():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "Empty message"}), 400

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ],
            temperature=0.3,
            max_tokens=300
        )

        raw_reply = completion.choices[0].message.content.strip()
        print("🧠 Raw Groq Response:", raw_reply)

        try:
            result = json.loads(raw_reply)
        except Exception:
            return jsonify({"intent": "UNKNOWN", "raw_reply": raw_reply}), 200

        # ===============================
        # Handle TASK intent
        # ===============================
        if result.get("intent") == "TASK":
            task_name = result.get("task_name", "Untitled Task")
            due_date = result.get("due_date")

            if not due_date:
                due_date = datetime.now(IST).isoformat()

            payload = {
                "parent": {"database_id": NOTION_DATABASE_ID},
                "properties": {
                    "Task Name": {"title": [{"text": {"content": task_name}}]},
                    "Status": {"select": {"name": result.get("status", "To Do")}},
                    "Avatar": {"select": {"name": result.get("avatar", "Producer")}},
                    "Due Date": {"date": {"start": due_date}}
                }
        }


            headers = {
                "Authorization": f"Bearer {NOTION_API_KEY}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json"
            }

            notion_resp = requests.post(
                "https://api.notion.com/v1/pages",
                headers=headers,
                json=payload
            )

            print("📝 Notion Response:", notion_resp.status_code, notion_resp.text)

            if notion_resp.status_code not in [200, 201]:
                return jsonify({
                    "error": "Failed to add to Notion",
                    "details": notion_resp.text
                }), 500

            return jsonify({
                "intent": "TASK",
                "status": "Task added to Notion",
                "task_name": task_name,
                "due_date": due_date
            }), 200

        # ===============================
        # Handle RESEARCH intent
        # ===============================
        elif result.get("intent") == "RESEARCH":
            return jsonify({
                "intent": "RESEARCH",
                "response": "I can handle your query soon!"
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
    app.run(host="0.0.0.0", port=10000)
