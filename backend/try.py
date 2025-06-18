from pymongo import MongoClient

client = MongoClient("")
print("Connection successfull")

db = client["hackathon_db"]  # your database name
print("DB Connection successfull")

collection = db["mcdonalds_customer_purchases"]  # your collection name
print("Collection Connection successfull")


for doc in collection.find():
    print(doc)
