# Third-party
from dotenv import load_dotenv

# Standard
import os
from pathlib import Path

load_dotenv('./.env')  # Load environment variables from .env

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / 'data'

# Define project directories
project = {
    'base': BASE,
}

# Define bot configuration
bot = {
    'token': os.getenv('BOT_TOKEN'),
}


# Define database configuration
database = {
    'host': os.getenv('DATABASE_HOST'),
    'port': os.getenv('DATABASE_PORT'),
    'user': os.getenv('DATABASE_USER'),
    'password': os.getenv('DATABASE_PASSWORD'),
}

# Define server configuration
server = {
    'host': os.getenv('PANEL_HOST'),
    'port': os.getenv('PANEL_PORT'),
    'secret_key': os.getenv('SECRET_KEY')
}