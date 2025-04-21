from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from typing import Dict, List, Optional, Any, Union
import json
import os

class BigQueryConnector:
    """
    A class for connecting to BigQuery, extracting schema information,
    and executing queries.
    """
    
    def __init__(self, credentials_path: Optional[str] = None, project_id: Optional[str] = None):
        """
        Initialize BigQuery connector with credentials
        
        Parameters:
        -----------
        credentials_path : str, optional
            Path to the service account credentials JSON file
        project_id : str, optional
            Google Cloud project ID (will be extracted from credentials if not provided)
        """
        self.credentials_path = credentials_path
        self.project_id = project_id
        self.client = None
        self.credentials = None
        self.schema_cache = {}
        
        # Try to initialize the client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize BigQuery client using credentials"""
        try:
            # If credentials path is provided, use it
            if self.credentials_path and os.path.exists(self.credentials_path):
                self.credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                if not self.project_id:
                    # Extract project_id from credentials
                    with open(self.credentials_path, 'r') as f:
                        cred_data = json.load(f)
                        self.project_id = cred_data.get('project_id')
                
                self.client = bigquery.Client(
                    credentials=self.credentials,
                    project=self.project_id
                )
            else:
                # Try default credentials
                self.client = bigquery.Client()
                self.project_id = self.client.project
        except Exception as e:
            print(f"Error initializing BigQuery client: {str(e)}")
            self.client = None
    
    def is_connected(self) -> bool:
        """Check if connection to BigQuery is established"""
        return self.client is not None
    
    def list_datasets(self) -> List[str]:
        """
        List all available datasets in the project
        
        Returns:
        --------
        List[str]
            List of dataset names
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to BigQuery")
        
        datasets = list(self.client.list_datasets())
        return [dataset.dataset_id for dataset in datasets]
    
    def list_tables(self, dataset_id: str) -> List[str]:
        """
        List all tables in a specific dataset
        
        Parameters:
        -----------
        dataset_id : str
            ID of the dataset
            
        Returns:
        --------
        List[str]
            List of table names
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to BigQuery")
        
        tables = list(self.client.list_tables(dataset_id))
        return [table.table_id for table in tables]
    
    def get_table_schema(self, dataset_id: str, table_id: str) -> List[Dict[str, Any]]:
        """
        Get schema information for a specific table
        
        Parameters:
        -----------
        dataset_id : str
            ID of the dataset
        table_id : str
            ID of the table
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of column information (name, type, description, etc.)
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to BigQuery")
        
        cache_key = f"{dataset_id}.{table_id}"
        if cache_key in self.schema_cache:
            return self.schema_cache[cache_key]
        
        # Get table reference
        table_ref = self.client.dataset(dataset_id).table(table_id)
        table = self.client.get_table(table_ref)
        
        # Extract schema information
        schema_info = []
        for field in table.schema:
            field_info = {
                'name': field.name,
                'type': field.field_type,
                'mode': field.mode,  # NULLABLE, REQUIRED, REPEATED
                'description': field.description or ""
            }
            schema_info.append(field_info)
        
        # Cache the result
        self.schema_cache[cache_key] = schema_info
        return schema_info
    
    def extract_all_schemas(self) -> Dict[str, Any]:
        """
        Extract schema information for all tables in all datasets
        
        Returns:
        --------
        Dict[str, Any]
            Nested dictionary with all schema information
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to BigQuery")
        
        all_schemas = {}
        datasets = self.list_datasets()
        
        for dataset_id in datasets:
            all_schemas[dataset_id] = {}
            tables = self.list_tables(dataset_id)
            
            for table_id in tables:
                try:
                    schema_info = self.get_table_schema(dataset_id, table_id)
                    all_schemas[dataset_id][table_id] = schema_info
                except Exception as e:
                    print(f"Error extracting schema for {dataset_id}.{table_id}: {str(e)}")
        
        return all_schemas
    
    def generate_context_from_schema(self) -> str:
        """
        Generate a text context from schema information for NL to SQL systems
        
        Returns:
        --------
        str
            Formatted context with schema information
        """
        all_schemas = self.extract_all_schemas()
        context_parts = []
        
        for dataset_id, tables in all_schemas.items():
            for table_id, schema_info in tables.items():
                table_desc = f"Table: {dataset_id}.{table_id}\nColumns:"
                columns = []
                
                for field in schema_info:
                    col_desc = f"- {field['name']} ({field['type']})"
                    if field['description']:
                        col_desc += f": {field['description']}"
                    columns.append(col_desc)
                
                table_desc += "\n" + "\n".join(columns) + "\n"
                context_parts.append(table_desc)
        
        return "\n\n".join(context_parts)
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame
        
        Parameters:
        -----------
        query : str
            SQL query to execute
            
        Returns:
        --------
        pd.DataFrame
            Query results as DataFrame
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to BigQuery")
        
        try:
            # Execute the query
            query_job = self.client.query(query)
            # Wait for the query to finish
            result = query_job.result()
            # Convert to DataFrame
            return result.to_dataframe()
        except Exception as e:
            print(f"Error executing query: {str(e)}")
            raise