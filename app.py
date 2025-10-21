import os
import json
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# --- INITIALIZATION ---

# 1. Initialize the Flask App
app = Flask(__name__)

# 2. Load Environment Variables
# This securely loads the API key from your .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is available
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

# 3. Configure the Gemini API
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash') # Using a fast and capable model

# 4. Load the Knowledge Base
# We load your website's content once when the server starts.
try:
    with open('knowledge_base.json', 'r') as f:
        knowledge_base_data = json.load(f)
    # Convert the JSON object to a string to use in the prompt
    knowledge_base_string = json.dumps(knowledge_base_data, indent=2)
except FileNotFoundError:
    raise FileNotFoundError("knowledge_base.json not found. Make sure it's in the same directory.")

# 5. Define the AI's "Brain" - The System Prompt
# This is the most important part. It's our instruction manual for Gemini.
# It tells the AI its name, its personality, and most importantly, to ONLY use the provided text (your website content).
SYSTEM_PROMPT = f"""
You are "Orech AI", a friendly, professional, and helpful AI assistant for Orech Rock Technologies.
Your sole purpose is to answer user questions based ONLY on the context provided below.
Do not use any external information or prior knowledge.
If the user asks a question that cannot be answered from the context, politely say something like, "I don't have information on that topic. I can help with questions about our services, mission, or how to contact us."
Keep your answers concise and directly related to the user's question.

---
CONTEXT:
{knowledge_base_string}
---
"""

# --- API ENDPOINT ---

# This is the URL our frontend will send messages to.
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the user's message from the request
        user_question = request.json.get('message')

        if not user_question:
            return jsonify({"error": "No message provided"}), 400

        # Create a new chat session with the system prompt
        chat_session = model.start_chat(
            history=[
                {'role': 'user', 'parts': [SYSTEM_PROMPT]},
                {'role': 'model', 'parts': ["Understood. I am Orech AI, and I will only use the provided context to answer questions. How can I help?"]}
            ]
        )
        
        # Send the user's question to the chat session and get the response
        response = chat_session.send_message(user_question)

        # Return the AI's response text to the frontend
        return jsonify({'response': response.text})

    except Exception as e:
        # Handle any potential errors gracefully
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# This line allows you to run the server by typing "python app.py" in the terminal
if __name__ == '__main__':
    app.run(debug=True, port=5000)
