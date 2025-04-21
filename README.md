# Natural Language to SQL Dashboard Generator

This application allows users to ask questions in natural language, which are then:
1. Converted to SQL queries for BigQuery
2. Executed against your BigQuery database
3. Results are analyzed based on data types
4. A comprehensive dashboard is automatically generated

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure BigQuery Authentication

You'll need to set up authentication credentials for BigQuery:

1. Create a service account in Google Cloud Console
2. Generate a JSON key file
3. Set the environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service_account_key.json"
   ```

### 3. Configure OpenAI API

This application uses OpenAI for natural language processing:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

## Running the Application

Run the Streamlit app:

```bash
streamlit run main.py
```

## Usage

1. Enter a question in natural language related to your BigQuery data
   - Example: "What were the top 5 products by sales last month?"
   - Example: "Show me the trend of user signups over the last year"

2. Click "Generate Answer"

3. The application will:
   - Show the generated SQL query
   - Display a summary of the results
   - Present a dashboard with relevant visualizations

## Project Structure

- `main.py`: Entry point for the Streamlit application
- `bigquery_connector.py`: Connects to BigQuery and fetches schema information
- `query_generator.py`: Converts natural language to SQL
- `data_analyzer.py`: Analyzes query results based on data types
- `dashboard_generator.py`: Creates visualizations and dashboards

## Customization

- To modify the visualizations, edit the `_generate_chart_configs` method in `data_analyzer.py`
- To adjust the dashboard layout, modify the `generate_dashboard` method in `dashboard_generator.py`
- To customize the SQL generation, edit the prompt in `query_generator.py`