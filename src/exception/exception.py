import sys


class NetworkSecurityException(Exception):
    """
    Custom exception for the Network Security.
    It captures the error message, filename, and line number of the exception.
    """

    def __init__(self, error_message: str):
        """
        Initializes the NetworkSecurityException.
        :param error_message: The error message for the exception.
        """
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_tb is None:
            raise ValueError(
                "This exception must be raised from within an except block."
            )

        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename
        self.error_message = f"Error occurred in [{self.filename}], in line [{self.lineno}]. Error message: [{str(error_message)}]"
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message
