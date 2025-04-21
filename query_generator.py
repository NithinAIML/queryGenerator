import openai
import os
from typing import Dict, Optional
import re

class QueryGenerator:
    """
    A class to generate SQL queries from natural language using LLMs.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                model: str = "gpt-4-turbo-preview"):
        """
        Initialize the query generator with OpenAI API credentials
        
        Parameters:
        -----------
        api_key : str, optional
            OpenAI API key (will use environment variable if not provided)
        model : str
            The LLM model to use for query generation
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize OpenAI client"""
        if self.api_key:
            openai.api_key = self.api_key
        else:
            print("Warning: No OpenAI API key provided. Using environment variable if available.")
    
    def generate_sql_query(self, 
                          user_question: str, 
                          schema_context: str) -> Dict[str, str]:
        """
        Generate a SQL query from a natural language question
        
        Parameters:
        -----------
        user_question : str
            The natural language question to convert to SQL
        schema_context : str
            The database schema information as context
            
        Returns:
        --------
        Dict[str, str]
            Dictionary with 'query' (SQL query) and 'explanation' (explanation of the query)
        """
        prompt = self._create_prompt(user_question, schema_context)
        
        try:
            # Call OpenAI API to generate SQL
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that translates natural language questions into BigQuery SQL queries. Only respond with the SQL query and a short explanation. Format your response as 'SQL: <query>\n\nExplanation: <explanation>'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more deterministic results
                max_tokens=1000
            )
            
            # Extract SQL and explanation from response
            sql_response = response.choices[0].message.content
            return self._parse_response(sql_response)
            
        except Exception as e:
            print(f"Error generating SQL query: {str(e)}")
            return {
                "query": "",
                "explanation": f"Error generating SQL query: {str(e)}"
            }
    
    def _create_prompt(self, user_question: str, schema_context: str) -> str:
        """
        Create a prompt for the LLM
        
        Parameters:
        -----------
        user_question : str
            The user's question
        schema_context : str
            Schema information
            
        Returns:
        --------
        str
            The formatted prompt
        """
        return f"""
I need to convert a natural language question into a BigQuery SQL query.

DATABASE SCHEMA INFORMATION:
{schema_context}

USER QUESTION:
{user_question}

Please generate a correct BigQuery SQL query that answers this question.
Your query should be optimized and follow best practices.
Make sure to include any necessary aggregations, grouping, filtering, or joining operations.
Only include tables and columns mentioned in the schema.
"""
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """
        Parse the response from the LLM to extract SQL query and explanation
        
        Parameters:
        -----------
        response : str
            The response from the LLM
            
        Returns:
        --------
        Dict[str, str]
            Parsed response with 'query' and 'explanation'
        """
        # Extract SQL query
        sql_match = re.search(r"SQL:?\s*(```sql\s*|\`\`\`\s*|\`)?(?P<query>[\s\S]*?)(```|\`)?(\n\n|\n)Explanation", response, re.IGNORECASE)
        if not sql_match:
            sql_match = re.search(r"(```sql\s*|\`\`\`\s*|\`)?(?P<query>SELECT[\s\S]*?)(```|\`)?(\n\n|\n)", response, re.IGNORECASE)
        
        if sql_match:
            sql_query = sql_match.group("query").strip()
        else:
            # Fallback: just try to find a SQL-looking query
            sql_query = re.search(r"SELECT[\s\S]*?(FROM[\s\S]*?)?(WHERE[\s\S]*?)?(GROUP BY[\s\S]*?)?(HAVING[\s\S]*?)?(ORDER BY[\s\S]*?)?(LIMIT \d+)?", response, re.IGNORECASE)
            sql_query = sql_query.group(0).strip() if sql_query else ""
        
        # Extract explanation
        explanation_match = re.search(r"Explanation:?\s*(?P<explanation>[\s\S]*?)($|```)", response, re.IGNORECASE)
        explanation = explanation_match.group("explanation").strip() if explanation_match else ""
        
        return {
            "query": sql_query,
            "explanation": explanation
        }