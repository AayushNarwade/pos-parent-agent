from flask import Flask, request, jsonify
import os, json
from dotenv import load_dotenv
from groq import Groq
from flask_cors import CORS

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-70b-versatile"


app = Flask(__name__)
CORS(app)  # allow calls from Streamlit

SYSTEM_PROMPT = """
You are the 'Parent Agent' for the "Present Operating System (POS)".
Classify the user's message into ONE of 3 intents:

TASK → To-do items, reminders, scheduling, habits, assignments
RESEARCH → Questions, info lookup, factual queries
UNKNOWN → Small talk, emotional text, unclear instruction

Return ONLY valid JSON in this exact structure:
{
  "intent": "TASK" | "RESEARCH" | "UNKNOWN",
  "data": "cleaned version of user request"
}

RULES:
- No explanations. JSON only.
- If message contains a verb+time/date/action → TASK.
- If message is a question → RESEARCH.
- If unclear, default to TASK.
"""

@app.route("/route", methods=["POST"])
def route_message():
    try:
        user_input = request.get_json().get("message", "")
        prompt = SYSTEM_PROMPT + "\n\nUser: " + user_input

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()
        try:
            result = json.loads(raw)
        except Exception:
            result = {"intent": "UNKNOWN", "data": user_input}
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
