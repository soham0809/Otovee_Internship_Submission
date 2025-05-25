import logging
import os
from datetime import datetime

# Logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with timestamp
LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

logging.basicConfig(
    filename=LOG_FILE,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO
)
