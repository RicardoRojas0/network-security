import logging
import os
from datetime import datetime

todays_date = datetime.now().strftime("%Y_%m_%d")
log_directory = os.path.join(os.getcwd(), "logs", todays_date)
os.makedirs(log_directory, exist_ok=True)

log_file_name = f"{datetime.now().strftime('%Y_%m_%d-%H:%M:%S')}.log"
log_file_path = os.path.join(log_directory, log_file_name)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
