from flask import Flask, render_template, request, jsonify
from query_generator import QueryGenerator
from bigquery_connector import BigQueryConnector
from data_analyzer import DataAnalyzer
from dashboard_generator import DashboardGenerator
import json

app = Flask(__name__)

# Initialize components
bq_connector = BigQueryConnector()
query_generator = QueryGenerator(bq_connector.get_schema_context())
data_analyzer = DataAnalyzer()
dashboard_generator = DashboardGenerator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    user_query = data.get('query', '')
    
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Generate SQL query from natural language
    sql_query = query_generator.generate_query(user_query)
    
    # Execute SQL query and get results
    results = bq_connector.execute_query(sql_query)
    
    # Analyze data
    analysis_results = data_analyzer.analyze(results)
    
    # Generate dashboard
    dashboard_html = dashboard_generator.generate_dashboard(analysis_results, user_query)
    
    response = {
        'sql_query': sql_query,
        'summary': analysis_results['summary'],
        'dashboard_html': dashboard_html
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)