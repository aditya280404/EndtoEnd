import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.logger import logging

def error_message_detail(error, error_detail):
    _, _, exc_tb = sys.exc_info()  # Fixed to use sys.exc_info() directly
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else 'Unknown'
    line_number = exc_tb.tb_lineno if exc_tb else 'Unknown'
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, line_number, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):  # Removed sys argument
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
