import os
from dotenv import load_dotenv
from db_helper import MongoDBHelper

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "stock_data")

def clear_data():
    db = MongoDBHelper()
    db.connect()
    
    try:
        coll = db.get_collection(COLLECTION_NAME)
        result = coll.delete_many({})
        print(f"Berhasil menghapus {result.deleted_count} dokumen dari koleksi '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"Terjadi kesalahan saat menghapus data: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    print(f"Mulai membersihkan data di koleksi '{COLLECTION_NAME}'...")
    clear_data()
    print("Pembersihan selesai.")
