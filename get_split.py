from pymongo import MongoClient
import re
import uuid
from dotenv import load_dotenv
import os


load_dotenv()


def get_mongo_splits_connection():
    # MongoDB connection details
    mongo_connection_string = os.getenv("MONGO_DB_CONNECTION_STRING")
    mongo_db= os.getenv("MONGO_DB")
    mongo_collection_splits= os.getenv("MONGO_COLLECTION_SPLITS")
    mongo_client = MongoClient(mongo_connection_string)
    db = mongo_client[mongo_db]
    splits_collection = db[mongo_collection_splits]
    return splits_collection


def get_split_to_detect_drift():
    splits_collection = get_mongo_splits_connection()
    result = splits_collection.find_one(
        {
            "is_complete": True,
            "$or": [
                {"used_for_drift_detection": None},
                {"used_for_drift_detection": False}
            ]
        },
        sort=[("date_created", 1)]
    )

    if result is None:
        print("No matching document found.")
        return None
    else:
        split_path = result.get("split_path")
        split_id = str(result.get("_id"))  # Convert ObjectId to string
        print(f"{split_path} {split_id}")
        return split_path, split_id

if __name__ == "__main__":
    split_info = get_split_to_detect_drift()
    if not split_info:
        exit(1)