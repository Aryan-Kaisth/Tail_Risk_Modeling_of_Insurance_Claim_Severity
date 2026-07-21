import os
import sys


def error_message_detail(error, error_detail=sys):
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        # Convert absolute path to relative path for cleaner log output
        file_name = os.path.relpath(exc_tb.tb_frame.f_code.co_filename)
        line_no = exc_tb.tb_lineno
        return f"Error occurred in python script [{file_name}] line number [{line_no}] error message [{error}]"

    # Fallback if raised outside an active except block
    return str(error)


class CustomException(Exception):
    """Custom wrapper to capture file name and line number automatically."""

    def __init__(self, error_message, error_detail=sys):
        super().__init__(error_message)
        # Enrich the error string with file & line context on creation
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message
