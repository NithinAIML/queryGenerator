from flask import Flask, render_template, request, jsonify
import sys
import os
import json
from dotenv import load_dotenv
from google.api_core.exceptions import Forbidden
import traceback

# Add the parent directory to sys.path to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from queryGenerator.analysis_chatbot import AnalysisChatbot

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize the chatbot with credentials from environment variables
try:
    chatbot = AnalysisChatbot(
        credentials_path=os.environ.get("BIGQUERY_CREDENTIALS_PATH"),
        project_id=os.environ.get("BIGQUERY_PROJECT_ID"),
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_api_endpoint=os.environ.get("OPENAI_API_ENDPOINT")
    )
    initialization_error = None
except Exception as e:
    chatbot = None
    initialization_error = str(e)
    print(f"Error initializing AnalysisChatbot: {e}")
    traceback.print_exc()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    """Process a user question and return analysis with visualizations."""
    if initialization_error:
        return jsonify({
            "status": "error", 
            "error": "Application initialization failed", 
            "details": initialization_error
        }), 500
    
    if not chatbot:
        return jsonify({
            "status": "error", 
            "error": "Chatbot not properly initialized"
        }), 500
        
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "error": "No JSON data provided"}), 400
            
        question = data.get('question', '')
        if not question:
            return jsonify({"status": "error", "error": "No question provided"}), 400
        
        # Process the question with the chatbot
        response = chatbot.process_question(question)
        
        # Verify the response is valid JSON-serializable
        try:
            # Test JSON serialization
            json.dumps(response)
            return jsonify(response)
        except (TypeError, OverflowError) as e:
            # Handle non-serializable objects in the response
            print(f"Non-serializable response: {e}")
            sanitized_response = {
                "status": "error",
                "error": "Response contained non-serializable data",
                "details": str(e)
            }
            return jsonify(sanitized_response), 500
            
    except Forbidden as e:
        if "VPC Service Controls" in str(e):
            return jsonify({
                "status": "error",
                "error": "VPC Service Controls restriction detected",
                "message": "This project requires connection from an authorized network. Please ensure you are connected to the correct VPN or network.",
                "details": str(e)
            }), 403
        else:
            return jsonify({"status": "error", "error": "Access forbidden", "message": str(e)}), 403
    except Exception as e:
        print(f"Error processing request: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "error": str(e) if str(e) else "An undefined error occurred",
            "type": type(e).__name__
        }), 500

@app.route('/refresh-schema', methods=['POST'])
def refresh_schema():
    """Force refresh of the schema context."""
    if not chatbot:
        return jsonify({"status": "error", "error": "Chatbot not properly initialized"}), 500
        
    try:
        chatbot.refresh_schema_context()
        return jsonify({"status": "success", "message": "Schema context refreshed successfully"})
    except Exception as e:
        print(f"Error refreshing schema: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "error": str(e) if str(e) else "An undefined error occurred while refreshing schema"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    status = {
        "status": "ok" if chatbot else "error",
        "bigquery_connected": chatbot.bq_connector.is_connected() if chatbot else False,
        "openai_initialized": hasattr(chatbot, 'openai_client') if chatbot else False,
        "initialization_error": initialization_error
    }
    return jsonify(status)

if __name__ == '__main__':
    # Check if the connection to BigQuery is established
    if chatbot and not chatbot.bq_connector.is_connected():
        print("Warning: BigQuery connection not established. Check your credentials.")
        print("If you're seeing VPC Service Controls errors, ensure you're connected to the appropriate VPN.")
    
    # Start the Flask app
    app.run(debug=True)