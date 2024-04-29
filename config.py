# Third-party
from dotenv import load_dotenv

# Standard
import os

load_dotenv('./.env')  # Load environment variables from .env

BASE = os.path.dirname(os.path.abspath(__file__))

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