import os
import pymongo
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()

class MongoDBHelper:
    def __init__(self):
        self.uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("DB_NAME", "finance_mlops")
        self.client = None
        self.db = None
    
    def connect(self):
        try:
            self.client = pymongo.MongoClient(self.uri)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            print(f"Successfully connected to MongoDB Atlas! Database: {self.db_name}")
        except ConnectionFailure as e:
            print(f"Could not connect to MongoDB: {e}")
            raise e
            
    def get_collection(self, collection_name):
        if self.db is None:
            self.connect()
        return self.db[collection_name]
    
    def insert_many(self, collection_name, records):
        collection = self.get_collection(collection_name)
        if records:
            result = collection.insert_many(records)
            print(f"Inserted {len(result.inserted_ids)} records into {collection_name}")
            return result
        return None
        
    def get_latest_date(self, collection_name, ticker):
        collection = self.get_collection(collection_name)
        # Assuming records have 'Date' and 'Ticker' fields
        latest_record = collection.find_one(
            {"Ticker": ticker},
            sort=[("Date", pymongo.DESCENDING)]
        )
        if latest_record:
            return latest_record['Date']
        return None
        
    def get_all_data(self, collection_name, ticker):
        collection = self.get_collection(collection_name)
        cursor = collection.find({"Ticker": ticker}).sort("Date", pymongo.ASCENDING)
        return list(cursor)
        
    def close(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")

if __name__ == "__main__":
    # Test initialization (will fail if MONGO_URI is not set, which is expected)
    helper = MongoDBHelper()
    print("MongoDBHelper initialized.")
