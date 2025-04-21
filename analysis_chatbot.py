import pandas as pd
from typing import Dict, List, Optional, Any, Union
import time
import json

from .bigquery_connector import BigQueryConnector
from .query_generator import QueryGenerator
from .visualizer import Visualizer

class AnalysisChatbot:
    """
    Main chatbot class that integrates BigQuery connection, SQL generation,
    data analysis, and visualization to respond to natural language questions.
    """
    
    def __init__(self, 
                credentials_path: Optional[str] = None, 
                project_id: Optional[str] = None,
                openai_api_key: Optional[str] = None):
        """
        Initialize the Analysis Chatbot with credentials
        
        Parameters:
        -----------
        credentials_path : str, optional
            Path to BigQuery service account credentials
        project_id : str, optional
            Google Cloud project ID
        openai_api_key : str, optional
            OpenAI API key for query generation
        """
        # Initialize BigQuery connector
        self.bq_connector = BigQueryConnector(credentials_path, project_id)
        
        # Initialize query generator
        self.query_generator = QueryGenerator(api_key=openai_api_key)
        
        # Initialize visualizer
        self.visualizer = Visualizer()
        
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
            
            # 4 & 5. Analyze the data and generate visualizations/dashboard
            dashboard_title = f"Analysis for: {user_question}"
            dashboard = self.visualizer.process_bigquery_results(
                data=data,
                query=query_result["query"],
                dashboard_title=dashboard_title
            )
            
            response["dashboard"] = dashboard
            
            # Add execution time
            response["execution_time_seconds"] = round(time.time() - start_time, 2)
            
            return response
            
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
        
        # Add visualization info
        if "dashboard" in response:
            dashboard = response["dashboard"]
            vis_count = len(dashboard.get("visualizations", []))
            formatted_parts.append(f"\nGenerated {vis_count} visualizations in dashboard: {dashboard['title']}")
        
        return "\n".join(formatted_parts)