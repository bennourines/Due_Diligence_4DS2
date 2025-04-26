# filepath: DeployTrial2/database/connection.py
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from core.config import settings
import logging
from pymongo import ASCENDING, DESCENDING # Import for index creation

logger = logging.getLogger(__name__)

class Database:
    client: AsyncIOMotorClient = None
    db: AsyncIOMotorDatabase = None

db_instance = Database()

async def connect_to_mongo():
    logger.info("Connecting to MongoDB...")
    try:
        db_instance.client = AsyncIOMotorClient(settings.MONGODB_URL)
        db_instance.db = db_instance.client[settings.MONGODB_DB_NAME]
        # Optional: Ping server to check connection
        await db_instance.client.admin.command('ping')
        logger.info(f"Successfully connected to MongoDB database: {settings.MONGODB_DB_NAME}")
        # Set up indexes after connection
        await setup_indexes(db_instance.db)
    except Exception as e:
        logger.error(f"Could not connect to MongoDB: {e}")
        # Depending on requirements, you might want the app to fail startup
        raise SystemExit(f"Failed to connect to MongoDB: {e}")


async def close_mongo_connection():
    if db_instance.client:
        logger.info("Closing MongoDB connection...")
        db_instance.client.close()
        logger.info("MongoDB connection closed.")

def get_database() -> AsyncIOMotorDatabase:
    if db_instance.db is None:
        # This case should ideally not happen if connect_to_mongo is called at startup
        # and raises SystemExit on failure.
        logger.critical("Database not initialized. Application might not have started correctly.")
        raise Exception("Database not initialized.")
    return db_instance.db

# Function to set up indexes
async def setup_indexes(db: AsyncIOMotorDatabase):
    logger.info("Setting up MongoDB indexes...")
    try:
        await db["users"].create_index("email", unique=True)
        await db["projects"].create_index([("user_id", ASCENDING)])
        await db["documents"].create_index([("project_id", ASCENDING), ("user_id", ASCENDING)])
        await db["history"].create_index([("project_id", ASCENDING), ("user_id", ASCENDING), ("timestamp", DESCENDING)])
        logger.info("MongoDB indexes set up successfully.")
    except Exception as e:
        logger.error(f"Error setting up MongoDB indexes: {e}", exc_info=True)
        # Decide if index creation failure should halt startup
