// Enhanced error handling for the query form
document.addEventListener('DOMContentLoaded', function() {
    const queryForm = document.getElementById('query-form');
    const questionInput = document.getElementById('question-input');
    const submitButton = document.getElementById('submit-button');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsContainer = document.getElementById('results-container');
    const errorContainer = document.getElementById('error-container');
    
    if (queryForm) {
        queryForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const question = questionInput.value.trim();
            
            if (!question) {
                showError('Please enter a question');
                return;
            }
            
            // Show loading state
            showLoading(true);
            clearResults();
            hideError();
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: question })
                });
                
                // Check if the response is OK
                if (!response.ok) {
                    const errorData = await response.json().catch(() => null);
                    throw new Error(
                        errorData?.error || `Server error: ${response.status} ${response.statusText}`
                    );
                }
                
                // Parse the response
                const data = await response.json().catch(err => {
                    console.error('Error parsing JSON response:', err);
                    throw new Error('Invalid response from server');
                });
                
                // Check for error in the response data
                if (data.status === 'error') {
                    throw new Error(data.error || data.message || 'Unknown error occurred');
                }
                
                // Display the results
                displayResults(data);
            } catch (error) {
                console.error('Query error:', error);
                showError(error.message || 'An undefined error occurred. Please check the console for details.');
            } finally {
                showLoading(false);
            }
        });
    }
    
    function showLoading(isLoading) {
        if (submitButton) submitButton.disabled = isLoading;
        if (loadingSpinner) loadingSpinner.style.display = isLoading ? 'block' : 'none';
    }
    
    function clearResults() {
        if (resultsContainer) resultsContainer.innerHTML = '';
    }
    
    function showError(message) {
        if (errorContainer) {
            errorContainer.textContent = message;
            errorContainer.style.display = 'block';
        }
    }
    
    function hideError() {
        if (errorContainer) {
            errorContainer.textContent = '';
            errorContainer.style.display = 'none';
        }
    }
    
    function displayResults(data) {
        if (resultsContainer) {
            resultsContainer.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        }
    }
});