import logging,os
from datetime import datetime

LOG_DIR="logs"
os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_PATH=os.path.join(LOG_DIR,LOG_FILE)

logging.basicConfig(
filename=LOG_PATH,
format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
level=logging.INFO)

def get_logger(name):
    return logging.getLogger(name)