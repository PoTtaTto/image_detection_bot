# Third-party
import sqlalchemy.exc
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker, joinedload

# Standard
from time import sleep
import traceback
from enum import Enum

# Project
import config as cf
from logger import database_logger
from .models import base, UserModel, SettingsModel


# Enum for different types of database connections
class Type(Enum):
    POSTGRESQL = f'postgresql+psycopg2://{cf.database["user"]}:{cf.database["password"]}@{cf.database["host"]}:{cf.database["port"]}'


class Database:
    """
    A class to interact with the database.

    Attributes:
    type_ (Type): The type of database connection.
    """

    # Private method to connect to the database
    def __connect_to_database(self, type_: Type):
        """
        Connect to the database using the specified type.

        Args:
        type_ (Type): The type of database connection.
        """
        while True:
            database_logger.warning('Connecting to database...')
            try:
                # Creating a database engine
                self.engine = create_engine(type_.value)
                self.session_maker = sessionmaker(bind=self.engine)
                # Creating tables defined in 'base' metadata
                base.metadata.create_all(self.engine)

                # __connect_inner_classes__ !DO NOT DELETE!


                database_logger.info('Connected to database')
                break
            except sqlalchemy.exc.OperationalError:
                # Handling database connection errors
                database_logger.error('Database error:\n' + traceback.format_exc())
                sleep(5.0)

    # Constructor to initialize the Database class
    def __init__(self, type_: Type):
        """
        Initialize the Database class with the specified type.

        Args:
        type_ (Type): The type of database connection.
        """
        self.__connect_to_database(type_=type_)



# Create an instance of the Database class with a PostgreSQL connection
db = Database(type_=Type.POSTGRESQL)