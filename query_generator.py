"""
SQL Query Generator using OpenAI LLMs
"""

from typing import Dict, Optional, List, Any
import json
import os
from .openai_client import OpenAIClient

class QueryGenerator:
    """
    Class for generating SQL queries from natural language questions using OpenAI
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                api_endpoint: Optional[str] = None,
                model: str = "gpt-3.5-turbo"):
        """
        Initialize the query generator
        
        Parameters:
        -----------
        api_key : str, optional
            OpenAI API key, will use OPENAI_API_KEY env var if not provided
        api_endpoint : str, optional
            OpenAI API endpoint URL, will use OPENAI_API_ENDPOINT env var if not provided
        model : str
            Model name to use for query generation
        """
        self.client = OpenAIClient(api_key=api_key, api_endpoint=api_endpoint)
        self.model = model
    
    def generate_sql_query(self, user_question: str, schema_context: str) -> Dict[str, str]:
        """
        Generate a SQL query from a natural language question
        
        Parameters:
        -----------
        user_question : str
            User's question in natural language
        schema_context : str
            BigQuery schema context
            
        Returns:
        --------
        Dict[str, str]
            Dictionary with query and explanation
        """
        # Create prompt with schema context and question
        messages = [
            {"role": "system", "content": f"""You are a SQL expert that generates BigQuery SQL queries based on user questions.
Use the following schema information to create your queries:

{schema_context}

Follow these rules:
1. Always generate standard BigQuery SQL.
2. Return valid SQL that will run directly in BigQuery.
3. Use only the tables and columns provided in the schema.
4. Format complex queries with appropriate line breaks and indentation.
5. Include appropriate JOINs based on the schema relationships.
6. Include a brief explanation of your query and key features.
7. For questions asking about trends over time, include appropriate date/time columns in your query.
8. If you cannot generate a query based on the provided information, explain why.
9. Respond with a JSON object containing two fields: 'query' with the SQL query and 'explanation' with your explanation."""},
            {"role": "user", "content": user_question}
        ]
        
        try:
            # Call the OpenAI API
            response = self.client.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.1,  # Low temperature for more deterministic outputs
                max_tokens=1500
            )
            
            # Extract the response text
            response_text = self.client.get_chat_completion_content(response)
            
            # Parse the JSON response
            try:
                # Try to extract JSON from the response
                # First check if the response is already valid JSON
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError:
                    # If not, try to extract JSON from markdown code blocks
                    if "```json" in response_text:
                        json_text = response_text.split("```json")[1].split("```")[0].strip()
                        result = json.loads(json_text)
                    elif "```" in response_text:
                        json_text = response_text.split("```")[1].strip()
                        result = json.loads(json_text)
                    else:
                        # If no code blocks, just try to find JSON-like content
                        start = response_text.find("{")
                        end = response_text.rfind("}") + 1
                        if start >= 0 and end > 0:
                            json_text = response_text[start:end]
                            result = json.loads(json_text)
                        else:
                            raise ValueError("Could not extract JSON from response")
                
                # Ensure the response has the expected structure
                if "query" not in result:
                    result["query"] = ""
                if "explanation" not in result:
                    result["explanation"] = "No explanation provided."
                
                return result
            
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing model response: {e}")
                # Return a fallback response if JSON parsing fails
                return {
                    "query": "",
                    "explanation": f"Error generating SQL query: {str(e)}. Response format was not as expected."
                }
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {
                "query": "",
                "explanation": f"Error generating SQL query: {str(e)}"
            }