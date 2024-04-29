# Third-party
from dotenv import load_dotenv

# Standard
import os
import pathlib

load_dotenv('./.env')  # Load environment variables from .env

BASE = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE / 'data'

# Create data folder if not exists
if not DATA_PATH.exists():
    DATA_PATH.mkdir()
    (DATA_PATH / 'images').mkdir()
    (DATA_PATH / 'model').mkdir()

# Create image folder if not exists
if not (DATA_PATH / 'images').exists():
    (DATA_PATH / 'images').mkdir()


# Define project directories
project = {
    'base': BASE,
}

# Define bot configuration
bot = {
    'token': os.getenv('BOT_TOKEN'),
}
