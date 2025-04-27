from pymongo import MongoClient

client = MongoClient("mongodb+srv://tripathikirtan9:EF0h5g9G7ne6Gc10@cluster0.jvlmy60.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
print("Connection successfull")

db = client["hackathon_db"]  # your database name
print("DB Connection successfull")

collection = db["mcdonalds_customer_purchases"]  # your collection name
print("Collection Connection successfull")


for doc in collection.find():
    print(doc)