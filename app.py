from flask import Flask, render_template, request, jsonify
import os
import json
from dotenv import load_dotenv

from .analysis_chatbot import AnalysisChatbot

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize the chatbot with credentials from environment variables
chatbot = AnalysisChatbot(
    credentials_path=os.environ.get("BIGQUERY_CREDENTIALS_PATH"),
    project_id=os.environ.get("BIGQUERY_PROJECT_ID"),
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Process a user question and return analysis with visualizations."""
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Process the question with the chatbot
    response = chatbot.process_question(question)
    
    return jsonify(response)

@app.route('/refresh-schema', methods=['POST'])
def refresh_schema():
    """Force refresh of the schema context."""
    try:
        chatbot.refresh_schema_context()
        return jsonify({"status": "success", "message": "Schema context refreshed successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Check if the connection to BigQuery is established
    if not chatbot.bq_connector.is_connected():
        print("Warning: BigQuery connection not established. Check your credentials.")
    
    # Start the Flask app
    app.run(debug=True)