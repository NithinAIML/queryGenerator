import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from plotly.io import to_html

class DashboardGenerator:
    def __init__(self):
        """Initialize the dashboard generator"""
        pass
        
    def generate_dashboard(self, analysis_results, query):
        """
        Generate dashboard HTML based on analysis results
        Args:
            analysis_results: Dictionary with analysis results and chart configs
            query: Original user query
        Returns:
            HTML string containing the dashboard
        """
        if "charts" not in analysis_results or not analysis_results["charts"]:
            return "<div class='alert alert-warning'>No visualizations could be generated for this query.</div>"
        
        df = analysis_results["dataframe"]
        
        # Starting the dashboard HTML
        dashboard_html = f"""
        <div class="dashboard-container">
            <h2>Dashboard for: {query}</h2>
            <ul class="nav nav-tabs" id="myTab" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" 
                    type="button" role="tab" aria-controls="overview" aria-selected="true">Overview</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="numerical-tab" data-bs-toggle="tab" data-bs-target="#numerical" 
                    type="button" role="tab" aria-controls="numerical" aria-selected="false">Numerical Analysis</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="categorical-tab" data-bs-toggle="tab" data-bs-target="#categorical" 
                    type="button" role="tab" aria-controls="categorical" aria-selected="false">Categorical Analysis</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="relationships-tab" data-bs-toggle="tab" data-bs-target="#relationships" 
                    type="button" role="tab" aria-controls="relationships" aria-selected="false">Relationships</button>
                </li>
            </ul>
            <div class="tab-content" id="myTabContent">
        """
        
        # Overview tab
        dashboard_html += """
            <div class="tab-pane fade show active" id="overview" role="tabpanel" aria-labelledby="overview-tab">
                <div class="mt-3">
                    <h3>Data Overview</h3>
                    <div class="table-responsive">
                        <table class="table table-striped table-sm">
                            <thead>
                                <tr>
        """
        
        # Add columns
        for col in df.columns:
            dashboard_html += f"<th>{col}</th>"
        
        dashboard_html += """
                                </tr>
                            </thead>
                            <tbody>
        """
        
        # Add rows (limit to 10)
        for _, row in df.head(10).iterrows():
            dashboard_html += "<tr>"
            for col in df.columns:
                dashboard_html += f"<td>{row[col]}</td>"
            dashboard_html += "</tr>"
            
        dashboard_html += """
                            </tbody>
                        </table>
                    </div>
                </div>
        """
        
        # Add basic statistics if numerical columns exist
        if analysis_results["data_types"]["numerical"]:
            dashboard_html += """
                <div class="mt-4">
                    <h3>Basic Statistics</h3>
                    <div class="table-responsive">
                        <table class="table table-striped table-sm">
                            <thead>
                                <tr>
                                    <th>Metric</th>
            """
            
            for col in analysis_results["data_types"]["numerical"]:
                dashboard_html += f"<th>{col}</th>"
                
            dashboard_html += """
                                </tr>
                            </thead>
                            <tbody>
            """
            
            # Add statistical metrics
            stats_metrics = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
            desc_stats = df[analysis_results["data_types"]["numerical"]].describe().to_dict()
            
            for metric in stats_metrics:
                dashboard_html += f"<tr><td>{metric}</td>"
                for col in analysis_results["data_types"]["numerical"]:
                    val = desc_stats[col][metric] if metric in desc_stats[col] else ''
                    dashboard_html += f"<td>{val:.4f}</td>" if isinstance(val, float) else f"<td>{val}</td>"
                dashboard_html += "</tr>"
                
            dashboard_html += """
                            </tbody>
                        </table>
                    </div>
                </div>
            """
            
        dashboard_html += "</div>"  # Close overview tab
        
        # Numerical Analysis tab
        dashboard_html += """
            <div class="tab-pane fade" id="numerical" role="tabpanel" aria-labelledby="numerical-tab">
        """
        
        numerical_charts = [c for c in analysis_results["charts"] 
                           if c["type"] in ["histogram", "box"] and "x" in c and c["x"] in analysis_results["data_types"]["numerical"]]
        
        if numerical_charts:
            for chart_config in numerical_charts[:6]:  # Limit to 6 charts
                chart_html = self._render_chart(chart_config, df)
                dashboard_html += f"""
                <div class="chart-container">
                    {chart_html}
                </div>
                """
        else:
            dashboard_html += """
                <div class="alert alert-info mt-3">
                    No numerical data to analyze.
                </div>
            """
            
        dashboard_html += "</div>"  # Close numerical tab
        
        # Categorical Analysis tab
        dashboard_html += """
            <div class="tab-pane fade" id="categorical" role="tabpanel" aria-labelledby="categorical-tab">
        """
        
        categorical_charts = [c for c in analysis_results["charts"] 
                             if c["type"] in ["bar", "pie"] and 
                             (("x" in c and c["x"] in analysis_results["data_types"]["categorical"]) or
                              ("values" in c and c["values"] in analysis_results["data_types"]["categorical"]))]
        
        if categorical_charts:
            for chart_config in categorical_charts[:6]:  # Limit to 6 charts
                chart_html = self._render_chart(chart_config, df)
                dashboard_html += f"""
                <div class="chart-container">
                    {chart_html}
                </div>
                """
        else:
            dashboard_html += """
                <div class="alert alert-info mt-3">
                    No categorical data to analyze.
                </div>
            """
            
        dashboard_html += "</div>"  # Close categorical tab
        
        # Relationships tab
        dashboard_html += """
            <div class="tab-pane fade" id="relationships" role="tabpanel" aria-labelledby="relationships-tab">
        """
        
        relationship_charts = [c for c in analysis_results["charts"] 
                              if c["type"] in ["scatter", "heatmap", "line"] or
                              (c["type"] == "bar" and "config" in c and c["config"].get("agg") == "mean")]
        
        if relationship_charts:
            for chart_config in relationship_charts[:6]:  # Limit to 6 charts
                chart_html = self._render_chart(chart_config, df)
                dashboard_html += f"""
                <div class="chart-container">
                    {chart_html}
                </div>
                """
        else:
            dashboard_html += """
                <div class="alert alert-info mt-3">
                    No relationship analysis available.
                </div>
            """
            
        dashboard_html += "</div>"  # Close relationships tab
        
        # Close containers
        dashboard_html += """
            </div>
        </div>
        """
        
        return dashboard_html
    
    def _render_chart(self, chart_config, df):
        """
        Render a chart based on configuration
        Args:
            chart_config: Dictionary with chart configuration
            df: Pandas DataFrame with data
        Returns:
            HTML representation of the chart
        """
        chart_type = chart_config["type"]
        title = chart_config.get("title", "Chart")
        
        try:
            if chart_type == "histogram":
                fig = px.histogram(
                    df, 
                    x=chart_config["x"],
                    title=title,
                    nbins=chart_config.get("config", {}).get("bins", 10),
                    opacity=0.7
                )
                
            elif chart_type == "box":
                fig = px.box(
                    df,
                    y=chart_config["y"],
                    title=title
                )
                
            elif chart_type == "bar":
                if chart_config.get("config", {}).get("agg") == "count":
                    # Simple count
                    fig = px.bar(
                        df[chart_config["x"]].value_counts().reset_index(),
                        x="index",
                        y=chart_config["x"],
                        title=title
                    )
                else:
                    # Aggregation (e.g., mean)
                    agg_func = chart_config.get("config", {}).get("agg", "mean")
                    agg_df = df.groupby(chart_config["x"])[chart_config["y"]].agg(agg_func).reset_index()
                    fig = px.bar(
                        agg_df,
                        x=chart_config["x"],
                        y=chart_config["y"],
                        title=title
                    )
                
            elif chart_type == "pie":
                fig = px.pie(
                    df,
                    names=chart_config["values"],
                    title=title
                )
                
            elif chart_type == "scatter":
                fig = px.scatter(
                    df,
                    x=chart_config["x"],
                    y=chart_config["y"],
                    title=title,
                    opacity=0.7
                )
                
            elif chart_type == "heatmap":
                columns = chart_config.get("config", {}).get("columns", [])
                if columns:
                    corr_matrix = df[columns].corr()
                    fig = px.imshow(
                        corr_matrix,
                        title=title,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1
                    )
                else:
                    return "<div class='alert alert-warning'>Cannot create heatmap: no columns specified</div>"
                
            elif chart_type == "line":
                # Sort by date if x is datetime
                if pd.api.types.is_datetime64_any_dtype(df[chart_config["x"]]):
                    plot_df = df.sort_values(chart_config["x"])
                else:
                    plot_df = df
                    
                fig = px.line(
                    plot_df,
                    x=chart_config["x"],
                    y=chart_config["y"],
                    title=title
                )
            else:
                return f"<div class='alert alert-warning'>Unknown chart type: {chart_type}</div>"
                
            # Make figures responsive
            fig.update_layout(
                autosize=True,
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            
            # Convert to HTML
            chart_html = to_html(fig, include_plotlyjs=False, full_html=False)
            return chart_html
                
        except Exception as e:
            return f"<div class='alert alert-danger'>Error generating {chart_type} chart: {str(e)}</div>"