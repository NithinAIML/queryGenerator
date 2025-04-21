import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union

class DataAnalyzer:
    """
    A class to analyze data and extract insights.
    Works with the Visualizer class to generate meaningful visualizations.
    """
    
    def __init__(self):
        pass
        
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the provided dataframe and return statistical insights
        
        Parameters:
        -----------
        data : pd.DataFrame
            The data to analyze
            
        Returns:
        --------
        Dict[str, Any]
            Dictionary of analysis results
        """
        if data is None or data.empty:
            return {"error": "No data available for analysis"}
            
        results = {}
        
        # Basic dataset info
        results["row_count"] = len(data)
        results["column_count"] = len(data.columns)
        
        # Get numerical columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        # Calculate basic statistics for numeric columns
        if numeric_cols:
            # Calculate summary statistics
            for col in numeric_cols:
                col_stats = {
                    f"{col}_mean": round(data[col].mean(), 2),
                    f"{col}_median": round(data[col].median(), 2),
                    f"{col}_min": round(data[col].min(), 2),
                    f"{col}_max": round(data[col].max(), 2),
                    f"{col}_std": round(data[col].std(), 2)
                }
                results.update(col_stats)
        
        # Get categorical columns and their value counts
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            if data[col].nunique() < 10:  # Only for columns with reasonable number of categories
                value_counts = data[col].value_counts().to_dict()
                # Format the dictionary values as a string
                results[f"{col}_counts"] = ', '.join([f"{k}: {v}" for k, v in value_counts.items()])
        
        # Check for missing values
        missing_values = data.isna().sum().to_dict()
        missing_values = {k: v for k, v in missing_values.items() if v > 0}
        if missing_values:
            results["missing_values"] = missing_values
        
        # Check for correlations between numeric columns
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().round(2)
            # Get the top 3 strongest correlations
            corr_pairs = []
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:  # Avoid duplicate pairs and self-correlations
                        corr_value = corr_matrix.loc[col1, col2]
                        if abs(corr_value) > 0.3:  # Only include meaningful correlations
                            corr_pairs.append((col1, col2, corr_value))
                            
            # Sort by absolute correlation value
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Add top correlations to results
            for i, (col1, col2, corr) in enumerate(corr_pairs[:3]):
                results[f"correlation_{i+1}"] = f"{col1} & {col2}: {corr}"
        
        return results
    
    def generate_insights(self, data: pd.DataFrame) -> List[str]:
        """
        Generate natural language insights about the data
        
        Parameters:
        -----------
        data : pd.DataFrame
            The data to analyze
            
        Returns:
        --------
        List[str]
            List of textual insights
        """
        if data is None or data.empty:
            return ["No data available for analysis"]
            
        insights = []
        row_count = len(data)
        col_count = len(data.columns)
        
        insights.append(f"The dataset contains {row_count} records with {col_count} columns.")
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            for col in numeric_cols:
                mean_val = data[col].mean()
                max_val = data[col].max()
                min_val = data[col].min()
                
                insights.append(f"The average {col} is {mean_val:.2f}, ranging from {min_val:.2f} to {max_val:.2f}.")
                
                # Check for outliers using IQR
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = len(data[(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))])
                
                if outlier_count > 0:
                    insights.append(f"There are {outlier_count} potential outliers in the {col} column.")
        
        # Get categorical columns
        cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            if data[col].nunique() < 10:
                top_category = data[col].value_counts().index[0]
                top_count = data[col].value_counts().iloc[0]
                top_percentage = (top_count / row_count) * 100
                
                insights.append(f"The most common {col} is '{top_category}', appearing in {top_percentage:.1f}% of the data.")
        
        # Check for correlations
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j and corr_matrix.loc[col1, col2] > 0.7:
                        corr_val = data[numeric_cols].corr().loc[col1, col2]
                        corr_type = "positive" if corr_val > 0 else "negative"
                        insights.append(f"There is a strong {corr_type} correlation of {corr_val:.2f} between {col1} and {col2}.")
        
        return insights