import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from typing import Dict, List, Union, Optional, Tuple, Any

class Visualizer:
    """
    A class to generate visualizations from data analysis results.
    Supports various types of plots including line, bar, scatter, histogram, etc.
    Can create dashboards from BigQuery results.
    """
    
    def __init__(self):
        # Set default styling
        self.style = 'whitegrid'
        self.context = 'talk'
        self.palette = 'viridis'
        self.figure_size = (10, 6)
        self._setup_styling()
    
    def _setup_styling(self):
        """Configure the default styling for plots"""
        sns.set_style(self.style)
        sns.set_context(self.context)
        plt.rcParams['figure.figsize'] = self.figure_size
    
    def create_visualization(self, 
                            data: pd.DataFrame, 
                            viz_type: str, 
                            x_column: Optional[str] = None,
                            y_column: Optional[str] = None,
                            title: str = "Data Visualization",
                            **kwargs) -> str:
        """
        Create a visualization based on the specified type and return as base64 image
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data to visualize
        viz_type : str
            Type of visualization (line, bar, scatter, hist, box, heatmap, etc.)
        x_column : str, optional
            Column name for x-axis
        y_column : str, optional
            Column name for y-axis
        title : str
            Title of the plot
        **kwargs : dict
            Additional parameters for the specific visualization
            
        Returns:
        --------
        str
            Base64 encoded image string for HTML embedding
        """
        plt.figure(figsize=self.figure_size)
        
        if viz_type.lower() == 'line':
            self._create_line_plot(data, x_column, y_column, title, **kwargs)
        elif viz_type.lower() == 'bar':
            self._create_bar_plot(data, x_column, y_column, title, **kwargs)
        elif viz_type.lower() == 'scatter':
            self._create_scatter_plot(data, x_column, y_column, title, **kwargs)
        elif viz_type.lower() == 'hist':
            self._create_histogram(data, x_column, title, **kwargs)
        elif viz_type.lower() == 'box':
            self._create_box_plot(data, x_column, y_column, title, **kwargs)
        elif viz_type.lower() == 'heatmap':
            self._create_heatmap(data, title, **kwargs)
        elif viz_type.lower() == 'pie':
            self._create_pie_chart(data, x_column, y_column, title, **kwargs)
        else:
            raise ValueError(f"Visualization type '{viz_type}' not supported")
        
        # Convert plot to base64 for HTML embedding
        return self._fig_to_base64()
    
    def _fig_to_base64(self) -> str:
        """Convert matplotlib figure to base64 encoded string for HTML embedding"""
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', bbox_inches='tight')
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        plt.close()
        return img_base64
    
    def _create_line_plot(self, data: pd.DataFrame, x_column: str, y_column: str, 
                         title: str, **kwargs):
        """Create a line plot"""
        sns.lineplot(data=data, x=x_column, y=y_column, **kwargs)
        plt.title(title)
        plt.tight_layout()
    
    def _create_bar_plot(self, data: pd.DataFrame, x_column: str, y_column: str, 
                        title: str, **kwargs):
        """Create a bar plot"""
        sns.barplot(data=data, x=x_column, y=y_column, **kwargs)
        plt.title(title)
        plt.xticks(rotation=45 if len(data) > 5 else 0)
        plt.tight_layout()
    
    def _create_scatter_plot(self, data: pd.DataFrame, x_column: str, y_column: str, 
                            title: str, **kwargs):
        """Create a scatter plot"""
        hue = kwargs.pop('hue', None)
        sns.scatterplot(data=data, x=x_column, y=y_column, hue=hue, **kwargs)
        plt.title(title)
        plt.tight_layout()
    
    def _create_histogram(self, data: pd.DataFrame, column: str, title: str, **kwargs):
        """Create a histogram"""
        bins = kwargs.pop('bins', 10)
        sns.histplot(data=data, x=column, bins=bins, **kwargs)
        plt.title(title)
        plt.tight_layout()
    
    def _create_box_plot(self, data: pd.DataFrame, x_column: str, y_column: str, 
                        title: str, **kwargs):
        """Create a box plot"""
        sns.boxplot(data=data, x=x_column, y=y_column, **kwargs)
        plt.title(title)
        plt.xticks(rotation=45 if len(data[x_column].unique()) > 5 else 0)
        plt.tight_layout()
    
    def _create_heatmap(self, data: pd.DataFrame, title: str, **kwargs):
        """Create a heatmap"""
        sns.heatmap(data, annot=kwargs.pop('annot', True), cmap=kwargs.pop('cmap', 'viridis'), **kwargs)
        plt.title(title)
        plt.tight_layout()
    
    def _create_pie_chart(self, data: pd.DataFrame, label_column: str, value_column: str, 
                         title: str, **kwargs):
        """Create a pie chart"""
        # Group data if needed
        if len(data) > 10:  # Limit pie slices for readability
            top_n = kwargs.pop('top_n', 5)
            grouped_data = data.groupby(label_column)[value_column].sum().nlargest(top_n)
            other_value = data.groupby(label_column)[value_column].sum().sum() - grouped_data.sum()
            
            # Add "Other" category if needed
            if other_value > 0:
                grouped_data = pd.concat([grouped_data, pd.Series({'Other': other_value})])
            
            labels = grouped_data.index
            values = grouped_data.values
        else:
            labels = data[label_column]
            values = data[value_column]
        
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, **kwargs)
        plt.axis('equal')
        plt.title(title)
        
    def create_multiple_visualizations(self, data: pd.DataFrame, 
                                     visualizations: List[Dict]) -> List[str]:
        """
        Create multiple visualizations based on configuration and return as list of base64 images
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data to visualize
        visualizations : List[Dict]
            List of visualization configurations
            Each dict should have keys: viz_type, x_column, y_column, title, etc.
            
        Returns:
        --------
        List[str]
            List of base64 encoded image strings for HTML embedding
        """
        results = []
        for viz_config in visualizations:
            viz_type = viz_config.pop('viz_type')
            x_column = viz_config.pop('x_column', None)
            y_column = viz_config.pop('y_column', None)
            title = viz_config.pop('title', f"{viz_type.capitalize()} Visualization")
            
            try:
                img_base64 = self.create_visualization(
                    data, viz_type, x_column, y_column, title, **viz_config
                )
                results.append(img_base64)
            except Exception as e:
                print(f"Error creating visualization: {str(e)}")
                continue
                
        return results
    
    def get_recommended_visualizations(self, data: pd.DataFrame) -> List[Dict]:
        """
        Recommend visualizations based on the data structure and content
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data to analyze
            
        Returns:
        --------
        List[Dict]
            List of recommended visualization configurations
        """
        recommendations = []
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
        
        # If we have datetime and numeric columns, suggest a line plot
        if datetime_columns and numeric_columns:
            recommendations.append({
                'viz_type': 'line',
                'x_column': datetime_columns[0],
                'y_column': numeric_columns[0],
                'title': f'Time Series of {numeric_columns[0]}'
            })
        
        # If we have categorical and numeric columns, suggest a bar plot and box plot
        if categorical_columns and numeric_columns:
            # Pick a categorical column with reasonable cardinality
            cat_col = None
            for col in categorical_columns:
                unique_vals = data[col].nunique()
                if 2 <= unique_vals <= 15:  # Not too many categories
                    cat_col = col
                    break
            
            if cat_col:
                recommendations.append({
                    'viz_type': 'bar',
                    'x_column': cat_col,
                    'y_column': numeric_columns[0],
                    'title': f'{numeric_columns[0]} by {cat_col}'
                })
                
                # Add a box plot to show distribution by category
                recommendations.append({
                    'viz_type': 'box',
                    'x_column': cat_col,
                    'y_column': numeric_columns[0],
                    'title': f'Distribution of {numeric_columns[0]} by {cat_col}'
                })
                
                # If there are multiple numeric columns, suggest grouped bar charts
                if len(numeric_columns) >= 2:
                    recommendations.append({
                        'viz_type': 'bar',
                        'x_column': cat_col,
                        'y_column': numeric_columns[1],
                        'title': f'{numeric_columns[1]} by {cat_col}'
                    })
        
        # If we have at least 2 numeric columns, suggest a scatter plot
        if len(numeric_columns) >= 2:
            recommendations.append({
                'viz_type': 'scatter',
                'x_column': numeric_columns[0],
                'y_column': numeric_columns[1],
                'title': f'Relationship between {numeric_columns[0]} and {numeric_columns[1]}'
            })
            
            # Also suggest a correlation heatmap if we have several numeric columns
            if len(numeric_columns) >= 3:
                recommendations.append({
                    'viz_type': 'heatmap',
                    'title': 'Correlation Heatmap',
                    'data_transform': 'correlation'
                })
        
        # For any numeric column, suggest a histogram
        if numeric_columns:
            for i, col in enumerate(numeric_columns[:3]):  # Limit to first 3 numeric columns
                recommendations.append({
                    'viz_type': 'hist',
                    'x_column': col,
                    'title': f'Distribution of {col}'
                })
        
        # For categorical columns, suggest pie charts if not too many categories
        for col in categorical_columns:
            unique_vals = data[col].nunique()
            if 2 <= unique_vals <= 10:  # Good for pie charts
                value_col = numeric_columns[0] if numeric_columns else None
                if value_col:
                    # If we have a numeric column, use it for values
                    recommendations.append({
                        'viz_type': 'pie',
                        'label_column': col,
                        'value_column': value_col,
                        'title': f'{value_col} Distribution by {col}'
                    })
                else:
                    # Otherwise use counts
                    recommendations.append({
                        'viz_type': 'pie',
                        'label_column': col,
                        'value_column': 'count',
                        'title': f'Distribution of {col}',
                        'data_transform': 'count'
                    })
        
        return recommendations
    
    def create_dashboard(self, data: pd.DataFrame, title: str = "Data Analysis Dashboard") -> Dict[str, Any]:
        """
        Create a complete dashboard from analysis results
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data from BigQuery results
        title : str
            Title for the dashboard
            
        Returns:
        --------
        Dict[str, Any]
            Dashboard configuration with visualizations and summary statistics
        """
        # Get data summary
        summary = self.get_data_summary(data)
        
        # Get recommended visualizations based on data
        viz_configs = self.get_recommended_visualizations(data)
        
        # Generate all visualizations
        visualizations = []
        for viz_config in viz_configs:
            # Handle data transformations if specified
            viz_data = data.copy()
            if 'data_transform' in viz_config:
                transform_type = viz_config.pop('data_transform')
                if transform_type == 'correlation':
                    viz_data = data.select_dtypes(include=['number']).corr()
                elif transform_type == 'count':
                    # Create a count-based dataframe for the specified column
                    col = viz_config.get('label_column')
                    if col:
                        count_data = data[col].value_counts().reset_index()
                        count_data.columns = [col, 'count']
                        viz_data = count_data
                        viz_config['value_column'] = 'count'
            
            try:
                # Create visualization
                viz_type = viz_config.pop('viz_type')
                x_column = viz_config.pop('x_column', None)
                y_column = viz_config.pop('y_column', None)
                title = viz_config.pop('title', f"{viz_type.capitalize()} Visualization")
                
                img_base64 = self.create_visualization(
                    viz_data, viz_type, x_column, y_column, title, **viz_config
                )
                
                # Store visualization info
                visualizations.append({
                    'type': viz_type,
                    'title': title,
                    'image': img_base64
                })
            except Exception as e:
                print(f"Error creating visualization: {str(e)}")
                continue
        
        # Compile dashboard
        dashboard = {
            'title': title,
            'summary': summary,
            'visualizations': visualizations,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return dashboard
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The data to analyze
            
        Returns:
        --------
        Dict[str, Any]
            Summary statistics including shape, column types, and basic descriptive statistics
        """
        numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_columns = [col for col in data.columns if pd.api.types.is_datetime64_any_dtype(data[col])]
        
        summary = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'column_types': {
                'numeric': len(numeric_columns),
                'categorical': len(categorical_columns),
                'datetime': len(datetime_columns)
            },
            'columns': {col: str(dtype) for col, dtype in data.dtypes.items()},
        }
        
        # Add descriptive statistics for numeric columns
        if numeric_columns:
            summary['numeric_stats'] = data[numeric_columns].describe().to_dict()
        
        # Add category counts for categorical columns (top 5 categories)
        if categorical_columns:
            summary['categorical_stats'] = {}
            for col in categorical_columns:
                top_values = data[col].value_counts().head(5).to_dict()
                total_unique = data[col].nunique()
                summary['categorical_stats'][col] = {
                    'unique_values': total_unique,
                    'top_values': top_values
                }
        
        # Add date range for datetime columns
        if datetime_columns:
            summary['datetime_stats'] = {}
            for col in datetime_columns:
                summary['datetime_stats'][col] = {
                    'min': data[col].min().isoformat() if not pd.isna(data[col].min()) else None,
                    'max': data[col].max().isoformat() if not pd.isna(data[col].max()) else None
                }
        
        return summary
        
    def process_bigquery_results(self, data: pd.DataFrame, query: str, 
                               dashboard_title: str = None) -> Dict[str, Any]:
        """
        Process BigQuery results to create a comprehensive dashboard
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The results from BigQuery
        query : str
            The SQL query that was executed
        dashboard_title : str, optional
            Title for the dashboard
            
        Returns:
        --------
        Dict[str, Any]
            Complete dashboard with visualizations, summary statistics, and SQL query info
        """
        if dashboard_title is None:
            dashboard_title = "Analysis of BigQuery Results"
            
        # Create dashboard
        dashboard = self.create_dashboard(data, title=dashboard_title)
        
        # Add query info
        dashboard['query_info'] = {
            'sql': query,
            'execution_time': pd.Timestamp.now().isoformat()
        }
        
        return dashboard