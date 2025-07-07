import sys
from src.logging.logger import logging


class NetworkSecurityException(Exception):
    def __init__(self, error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()

        self.lineno = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error ocurred in [{self.filename}], in line [{self.lineno}]. Error message: [{str(self.error_message)}]"


if __name__ == "__main__":
    try:
        logging.info("Entered the try block")
        a = 1 / 0
        print("This will not be printed")
    except Exception as e:
        raise NetworkSecurityException(error_message=e, error_details=sys)
