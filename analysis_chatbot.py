import pandas as pd
from typing import Dict, List, Optional, Any, Union
import time
import json
import requests
import urllib3.exceptions
import socket
from functools import wraps

from .bigquery_connector import BigQueryConnector
from .query_generator import QueryGenerator
from .visualizer import Visualizer
from .openai_client import OpenAIClient

def retry_on_connection_error(max_retries=3, delay=2):
    """Decorator to retry function calls on connection errors"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.ConnectionError, 
                        urllib3.exceptions.NewConnectionError,
                        urllib3.exceptions.MaxRetryError,
                        socket.gaierror) as e:
                    retries += 1
                    if retries < max_retries:
                        print(f"Connection error: {str(e)}. Retrying in {delay} seconds... ({retries}/{max_retries})")
                        time.sleep(delay)
                    else:
                        print(f"Connection error: {str(e)}. Max retries exceeded.")
                        raise ConnectionError(f"Error connecting to OpenAI API: {str(e)}. Please check your network connection and DNS settings.")
            return func(*args, **kwargs)
        return wrapper
    return decorator

class AnalysisChatbot:
    """
    Main chatbot class that integrates BigQuery connection, SQL generation,
    data analysis, and visualization to respond to natural language questions.
    """
    
    def __init__(self, 
                credentials_path: Optional[str] = None, 
                project_id: Optional[str] = None,
                openai_api_key: Optional[str] = None,
                openai_api_endpoint: Optional[str] = None,
                model: str = "gpt-3.5-turbo"):
        """
        Initialize the Analysis Chatbot with credentials
        
        Parameters:
        -----------
        credentials_path : str, optional
            Path to BigQuery service account credentials
        project_id : str, optional
            Google Cloud project ID
        openai_api_key : str, optional
            OpenAI API key for query generation and analysis
        openai_api_endpoint : str, optional
            OpenAI API endpoint URL
        model : str, optional
            OpenAI model to use
        """
        # Initialize BigQuery connector
        self.bq_connector = BigQueryConnector(credentials_path, project_id)
        
        # Initialize query generator with API key and endpoint URL
        self.query_generator = QueryGenerator(
            api_key=openai_api_key, 
            api_endpoint=openai_api_endpoint,
            model=model
        )
        
        # Initialize OpenAI client for analysis
        self.openai_client = OpenAIClient(
            api_key=openai_api_key,
            api_endpoint=openai_api_endpoint
        )
        
        # Initialize visualizer
        self.visualizer = Visualizer()
        
        # Store model name
        self.model = model
        
        # Store schema context for reuse
        self._schema_context = None
        
    def refresh_schema_context(self) -> str:
        """
        Refresh the schema context from BigQuery
        
        Returns:
        --------
        str
            The generated schema context
        """
        if not self.bq_connector.is_connected():
            raise ConnectionError("Not connected to BigQuery")
        
        self._schema_context = self.bq_connector.generate_context_from_schema()
        return self._schema_context
    
    def get_schema_context(self) -> str:
        """
        Get the cached schema context or generate a new one
        
        Returns:
        --------
        str
            The schema context
        """
        if not self._schema_context:
            return self.refresh_schema_context()
        return self._schema_context
    
    def check_openai_connectivity(self):
        """Check connectivity to OpenAI API"""
        try:
            # Simple DNS resolution test
            socket.gethostbyname('api.openai.com')
            return True
        except socket.gaierror:
            return False
    
    @retry_on_connection_error()
    def analyze_results(self, question: str, query_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze query results and generate insights
        
        Parameters:
        -----------
        question : str
            The original user question
        query_results : pd.DataFrame
            The query results to analyze
            
        Returns:
        --------
        Dict[str, Any]
            Analysis results with insights and visualization suggestions
        """
        # Convert DataFrame to string representation for the prompt
        data_sample = query_results.head(10).to_string()
        data_description = query_results.describe().to_string()
        
        column_info = []
        for col in query_results.columns:
            dtype = str(query_results[col].dtype)
            sample = str(query_results[col].iloc[0]) if len(query_results) > 0 else "N/A"
            column_info.append(f"- {col} ({dtype}): e.g., {sample}")
        
        column_details = "\n".join(column_info)
        
        # Create the prompt
        messages = [
            {"role": "system", "content": f"""You are a data analysis expert. Analyze the given query results and provide insights.
The data is the result of a SQL query executed against a BigQuery database.

Data summary:
{data_description}

Columns:
{column_details}

Follow these rules:
1. Provide detailed insights based on the data patterns.
2. Suggest visualizations that would best represent the data.
3. Keep explanations clear and concise.
4. Focus on answering the original question.
5. Respond with a JSON object containing: 'insights' (list of key findings) and 'visualizations' (list of suggested charts with explanations).
"""},
            {"role": "user", "content": f"""Original question: {question}

Here's a sample of the query results:
{data_sample}

Please analyze these results and provide insights relevant to the question."""}
        ]
        
        try:
            # Call OpenAI API
            response = self.openai_client.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.5,
                max_tokens=1000
            )
            
            # Extract the response content
            response_text = self.openai_client.get_chat_completion_content(response)
            
            # Parse the JSON response
            try:
                # Try to extract JSON from the response
                result = {}
                try:
                    # Try parsing directly
                    result = json.loads(response_text)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown
                    if "```json" in response_text:
                        json_text = response_text.split("```json")[1].split("```")[0].strip()
                        result = json.loads(json_text)
                    elif "```" in response_text:
                        json_text = response_text.split("```")[1].split("```")[0].strip()
                        try:
                            result = json.loads(json_text)
                        except json.JSONDecodeError:
                            pass
                
                # If JSON extraction failed, use the full text as insights
                if not result:
                    result = {
                        "insights": [response_text],
                        "visualizations": []
                    }
                
                # Ensure the result has the expected structure
                if "insights" not in result:
                    result["insights"] = ["No specific insights found."]
                if "visualizations" not in result:
                    result["visualizations"] = []
                
                return result
            
            except Exception as e:
                print(f"Error parsing analysis response: {e}")
                return {
                    "insights": [f"Error analyzing results: {str(e)}"],
                    "visualizations": []
                }
                
        except Exception as e:
            print(f"Error performing analysis: {e}")
            return {
                "insights": [f"Error analyzing results: {str(e)}"],
                "visualizations": []
            }
    
    def process_question(self, user_question: str) -> Dict[str, Any]:
        """
        Process a natural language question and generate response with analysis
        
        Parameters:
        -----------
        user_question : str
            The user's question in natural language
            
        Returns:
        --------
        Dict[str, Any]
            Complete response including answer, SQL query, and dashboard
        """
        start_time = time.time()
        response = {
            "question": user_question,
            "status": "success",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        try:
            # Check OpenAI connectivity before proceeding
            if not self.check_openai_connectivity():
                return {
                    "status": "error",
                    "message": "Cannot connect to OpenAI API. Please check your network connection and DNS settings.",
                    "details": "Failed to resolve api.openai.com - DNS resolution failed"
                }
            
            # 1. Get schema context
            schema_context = self.get_schema_context()
            
            # 2. Generate SQL query from natural language
            query_result = self.query_generator.generate_sql_query(
                user_question=user_question,
                schema_context=schema_context
            )
            
            response["sql_query"] = query_result["query"]
            response["query_explanation"] = query_result["explanation"]
            
            if not query_result["query"]:
                response["status"] = "error"
                response["error"] = "Failed to generate SQL query"
                return response
            
            # 3. Execute the query against BigQuery
            data = self.bq_connector.execute_query(query_result["query"])
            response["row_count"] = len(data)
            
            # 4. Analyze the data
            analysis_results = self.analyze_results(user_question, data)
            response["insights"] = analysis_results["insights"]
            
            # 5. Generate visualizations/dashboard
            dashboard_title = f"Analysis for: {user_question}"
            dashboard = self.visualizer.process_bigquery_results(
                data=data,
                query=query_result["query"],
                dashboard_title=dashboard_title,
                visualization_suggestions=analysis_results.get("visualizations", [])
            )
            
            response["dashboard"] = dashboard
            
            # Add execution time
            response["execution_time_seconds"] = round(time.time() - start_time, 2)
            
            return response
            
        except ConnectionError as e:
            return {
                "status": "error",
                "message": str(e),
                "details": "Network connectivity issue with OpenAI API"
            }
        except Exception as e:
            response["status"] = "error"
            response["error"] = str(e)
            return response
    
    def format_response_for_display(self, response: Dict[str, Any]) -> str:
        """
        Format the response for display to the user
        
        Parameters:
        -----------
        response : Dict[str, Any]
            The full response from process_question
            
        Returns:
        --------
        str
            Formatted response as a string
        """
        if response["status"] == "error":
            return f"Error: {response['error']}"
        
        formatted_parts = [
            f"Question: {response['question']}",
            "\nSQL Query:",
            f"```sql\n{response['sql_query']}\n```",
            f"\nExplanation: {response['query_explanation']}",
            f"\nRows returned: {response['row_count']}",
            f"\nExecution time: {response['execution_time_seconds']} seconds"
        ]
        
        # Add insights
        if "insights" in response:
            formatted_parts.append("\nInsights:")
            for i, insight in enumerate(response["insights"], 1):
                formatted_parts.append(f"{i}. {insight}")
        
        # Add visualization info
        if "dashboard" in response:
            dashboard = response["dashboard"]
            vis_count = len(dashboard.get("visualizations", []))
            formatted_parts.append(f"\nGenerated {vis_count} visualizations in dashboard: {dashboard['title']}")
        
        return "\n".join(formatted_parts)