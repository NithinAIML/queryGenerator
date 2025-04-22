"""
Error Handler for QueryGenerator application
Provides better error logging, tracking, and diagnosis
"""

import sys
import traceback
import logging
import json
import inspect
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("queryGenerator_errors.log"),
        logging.StreamHandler()
    ]
)

class ErrorHandler:
    """
    Handles, logs, and formats errors throughout the application
    """
    
    def __init__(self, logger_name: str = "queryGenerator"):
        """Initialize the error handler with a specific logger"""
        self.logger = logging.getLogger(logger_name)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error with context information
        
        Parameters:
        -----------
        error : Exception
            The exception object
        context : Dict[str, Any], optional
            Additional context information about where the error occurred
        """
        # Get calling function and line number
        frame = inspect.currentframe().f_back
        func_name = frame.f_code.co_name if frame else "unknown_function"
        line_no = frame.f_lineno if frame else 0
        
        # Log the error with traceback
        self.logger.error(
            f"Error in {func_name} (line {line_no}): {str(error) or 'undefined error'}"
        )
        
        # Log the traceback
        self.logger.error("".join(traceback.format_tb(error.__traceback__)))
        
        # Log the context if provided
        if context:
            self.logger.error(f"Context: {json.dumps(context, default=str)}")
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error, log it, and return a formatted error response
        
        Parameters:
        -----------
        error : Exception
            The exception object
        context : Dict[str, Any], optional
            Additional context information about where the error occurred
            
        Returns:
        --------
        Dict[str, Any]
            Formatted error response for the API
        """
        # Log the error
        self.log_error(error, context)
        
        # Extract meaningful information from the error
        error_type = type(error).__name__
        error_msg = str(error)
        
        # Handling empty error messages
        if not error_msg:
            if error_type == "TypeError":
                error_msg = "A type error occurred, possibly involving undefined values"
            elif error_type == "AttributeError":
                error_msg = "An attribute error occurred, possibly accessing undefined property"
            else:
                error_msg = "An undefined error occurred"
        
        # Create a response object
        response = {
            "status": "error",
            "error": error_msg,
            "error_type": error_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add context if provided
        if context:
            response["context"] = {k: str(v) for k, v in context.items()}
            
        return response
    
    def format_api_error(self, error: Exception, status_code: int = 500) -> Tuple[Dict[str, Any], int]:
        """
        Format an error for API responses
        
        Parameters:
        -----------
        error : Exception
            The exception object
        status_code : int, optional
            HTTP status code to return, defaults to 500
            
        Returns:
        --------
        Tuple[Dict[str, Any], int]
            Tuple containing the error response and HTTP status code
        """
        error_response = self.handle_error(error)
        return error_response, status_code
        
    def diagnose_undefined_error(self) -> str:
        """
        Diagnose potential causes of undefined errors
        
        Returns:
        --------
        str
            Diagnosis message
        """
        # Check Python version
        python_ver = sys.version
        
        # Check for common issues that might cause undefined errors
        diagnosis = [
            f"Python version: {python_ver}",
            "Potential causes of 'undefined' errors:",
            "- Missing environment variables or configuration values",
            "- Accessing properties of None objects",
            "- JSON serialization issues with non-serializable objects",
            "- Network connectivity problems with external services"
        ]
        
        return "\n".join(diagnosis)

# Create a global instance of the error handler
error_handler = ErrorHandler()