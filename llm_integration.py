import os
import requests
import json
from typing import Dict, List, Any, Optional, Union

class LLMIntegration:
    """
    A class to handle integration with OpenAI's API for natural language processing tasks.
    """
    
    def __init__(self, api_key: Optional[str] = None, api_endpoint: Optional[str] = None):
        """
        Initialize the LLM integration with API credentials.
        
        Parameters:
        -----------
        api_key : str, optional
            The OpenAI API key. If not provided, will try to get from environment variable OPENAI_API_KEY
        api_endpoint : str, optional
            The OpenAI API endpoint. If not provided, will try to get from environment variable OPENAI_API_ENDPOINT
        """
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.api_endpoint = api_endpoint or os.environ.get('OPENAI_API_ENDPOINT')
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
            
        if not self.api_endpoint:
            raise ValueError("OpenAI API endpoint not found. Please provide it or set OPENAI_API_ENDPOINT environment variable.")
    
    def get_visualization_recommendations(self, data_description: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get visualization recommendations based on data description using LLM.
        
        Parameters:
        -----------
        data_description : Dict[str, Any]
            Description of the data including columns, types, and summary statistics
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of recommended visualization configurations
        """
        prompt = self._create_viz_recommendation_prompt(data_description)
        response = self._call_openai_api(prompt)
        
        # Parse recommendations from response
        try:
            recommendations = json.loads(response)
            return recommendations
        except json.JSONDecodeError:
            # Fallback parsing if response is not valid JSON
            return self._parse_text_recommendations(response)
    
    def generate_insights(self, data_summary: Dict[str, Any], visualizations: List[Dict[str, Any]]) -> List[str]:
        """
        Generate textual insights based on data summary and visualizations.
        
        Parameters:
        -----------
        data_summary : Dict[str, Any]
            Summary statistics of the data
        visualizations : List[Dict[str, Any]]
            List of visualization results and configurations
            
        Returns:
        --------
        List[str]
            List of textual insights about the data
        """
        prompt = self._create_insights_prompt(data_summary, visualizations)
        response = self._call_openai_api(prompt)
        
        # Parse insights from response
        insights = response.split('\n')
        insights = [insight.strip() for insight in insights if insight.strip()]
        return insights
    
    def explain_sql_query(self, query: str) -> str:
        """
        Generate a natural language explanation of an SQL query.
        
        Parameters:
        -----------
        query : str
            The SQL query to explain
            
        Returns:
        --------
        str
            Natural language explanation of the query
        """
        prompt = f"Please explain the following SQL query in simple terms:\n\n{query}"
        return self._call_openai_api(prompt)
    
    def suggest_follow_up_queries(self, original_query: str, data_summary: Dict[str, Any]) -> List[str]:
        """
        Suggest follow-up SQL queries based on the original query and data summary.
        
        Parameters:
        -----------
        original_query : str
            The original SQL query
        data_summary : Dict[str, Any]
            Summary of the data returned by the original query
            
        Returns:
        --------
        List[str]
            List of suggested follow-up SQL queries
        """
        prompt = f"""Based on this original SQL query and the data summary it produced,
        suggest 3 follow-up SQL queries that could provide additional insights:
        
        ORIGINAL QUERY:
        {original_query}
        
        DATA SUMMARY:
        {json.dumps(data_summary, indent=2)}
        
        Return just the SQL queries without additional text, one query per line.
        """
        
        response = self._call_openai_api(prompt)
        
        # Parse queries from response
        queries = response.split('\n')
        queries = [q.strip() for q in queries if q.strip() and not q.startswith('#')]
        return queries
    
    def _call_openai_api(self, prompt: str) -> str:
        """
        Call the OpenAI API with the given prompt.
        
        Parameters:
        -----------
        prompt : str
            The prompt to send to the API
            
        Returns:
        --------
        str
            The response from the API
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "gpt-4",  # Can be configured based on needs
            "messages": [
                {"role": "system", "content": "You are a data analysis assistant that helps with visualization and insights."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,  # Lower temperature for more deterministic results
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error calling OpenAI API: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            return "Error: Failed to get response from OpenAI API"
    
    def _create_viz_recommendation_prompt(self, data_description: Dict[str, Any]) -> str:
        """
        Create a prompt for visualization recommendations based on data description.
        """
        return f"""Based on the following data description, recommend visualizations that would be appropriate:
        
        DATA DESCRIPTION:
        {json.dumps(data_description, indent=2)}
        
        Return a JSON array of visualization configurations. Each configuration should be a JSON object with at least:
        - viz_type: type of visualization (line, bar, scatter, hist, box, heatmap, pie)
        - x_column: column for x-axis (if applicable)
        - y_column: column for y-axis (if applicable)
        - title: suggested title for the visualization
        
        Example:
        [
            {{
                "viz_type": "bar",
                "x_column": "category",
                "y_column": "sales",
                "title": "Sales by Category"
            }},
            {{
                "viz_type": "line",
                "x_column": "date",
                "y_column": "price",
                "title": "Price Trends Over Time"
            }}
        ]
        """
    
    def _create_insights_prompt(self, data_summary: Dict[str, Any], visualizations: List[Dict[str, Any]]) -> str:
        """
        Create a prompt for generating insights based on data summary and visualizations.
        """
        viz_descriptions = []
        for i, viz in enumerate(visualizations):
            viz_descriptions.append(f"Visualization {i+1}: {viz.get('title', 'Untitled')} (Type: {viz.get('type', 'unknown')})")
        
        viz_text = "\n".join(viz_descriptions)
        
        return f"""Based on the following data summary and visualizations, provide key insights about the data:
        
        DATA SUMMARY:
        {json.dumps(data_summary, indent=2)}
        
        VISUALIZATIONS:
        {viz_text}
        
        List 3-5 clear insights that can be drawn from this data. Each insight should be concise and data-driven.
        """
    
    def _parse_text_recommendations(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse visualization recommendations from text when JSON parsing fails.
        A fallback method for handling unstructured responses.
        
        Parameters:
        -----------
        text : str
            The text response to parse
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of visualization configurations
        """
        recommendations = []
        
        # Look for common patterns in the response
        if "line" in text.lower():
            recommendations.append({
                "viz_type": "line",
                "title": "Time Series Visualization"
            })
        
        if "bar" in text.lower():
            recommendations.append({
                "viz_type": "bar",
                "title": "Bar Chart Visualization"
            })
            
        if "scatter" in text.lower():
            recommendations.append({
                "viz_type": "scatter",
                "title": "Scatter Plot Visualization"
            })
            
        if "histogram" in text.lower() or "hist" in text.lower():
            recommendations.append({
                "viz_type": "hist",
                "title": "Distribution Histogram"
            })
            
        if "heatmap" in text.lower():
            recommendations.append({
                "viz_type": "heatmap",
                "title": "Correlation Heatmap"
            })
            
        # If we couldn't extract any recommendations, add a default one
        if not recommendations:
            recommendations.append({
                "viz_type": "bar",
                "title": "Default Visualization"
            })
            
        return recommendations