# database.py
from motor.motor_asyncio import AsyncIOMotorClient
import os

async def get_mongodb_connection():
    """Get MongoDB connection"""
    client = AsyncIOMotorClient(os.getenv("MONGODB_URL", "mongodb+srv://Feriel:Feriel@cluster0.81oai.mongodb.net/"))
    return client.crypto_due_diligence

async def store_conversation(db, conversation):
    """Store conversation in MongoDB"""
    await db.conversations.insert_one(conversation)

