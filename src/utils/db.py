import os
import sys
import pandas as pd
import psycopg
from psycopg import Connection
from dotenv import load_dotenv

from src.exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

load_dotenv()


def get_connection() -> Connection:
    try:
        logger.info("Attempting to connect to PostgreSQL database.")

        conn = psycopg.connect(
            host=os.environ["DB_HOST"],
            port=int(os.environ.get("DB_PORT", 5432)),
            dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASS"],
        )

        logger.info("Successfully connected to PostgreSQL database.")
        return conn

    # Missing env vars in .env file
    except KeyError as e:
        logger.error(f"Missing environment variable: {e}")
        raise CustomException(e, sys)

    # Database server down or unreachable network
    except psycopg.OperationalError as e:
        logger.error(f"Failed to connect to PostgreSQL server: {e}")
        raise CustomException(e, sys)

    except Exception as e:
        logger.error(f"Unexpected error while creating database connection: {e}")
        raise CustomException(e, sys)


def fetch_dataframe(query: str) -> pd.DataFrame:
    try:
        logger.info("Executing SQL query.")

        with get_connection() as conn:
            df = pd.read_sql_query(query, conn)

        logger.info(f"Query executed successfully. Shape: {df.shape}")
        return df

    # Specific DB errors must be caught before parent ProgrammingError
    except psycopg.errors.UndefinedColumn as e:
        logger.error(f"Undefined Column: {e}")
        raise CustomException(e, sys)

    except psycopg.errors.UndefinedTable as e:
        logger.error(f"Undefined Table: {e}")
        raise CustomException(e, sys)

    # Bad SQL syntax or invalid queries
    except psycopg.ProgrammingError as e:
        logger.error(f"Invalid SQL query structure or syntax error: {e}")
        raise CustomException(e, sys)

    except Exception as e:
        logger.error(f"Unexpected error while fetching dataframe: {e}")
        raise CustomException(e, sys)