import os
import json
from openai import OpenAI
import psycopg2
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_reviews():
    """Load reviews from JSON file"""
    with open('winemag-data-130k-v2.json', 'r') as file:
        return json.load(file)

def generate_embeddings(batch_size=300):
    # Connect to Postgres
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="saurabhkamboj",
        database="wine_reviews"
    )
    cursor = conn.cursor()

    try:
        reviews = load_reviews()
        
        # Process chapters in batches
        for index1 in range(0, len(reviews), batch_size):
            batch = reviews[index1:index1 + batch_size]
            
            # Prepare batch data
            batch_contents = []
            batch_metadata = []
            
            for review in batch:
                # Create content string combining book title and chapter title
                content = f"""
                    Title: {review['title']}
                    Country: {review['country']}
                    Province: {review['province']}
                    Variety: {review['variety']}
                    Description: {review['description']}
                    """
                batch_contents.append(content)
                batch_metadata.append({
                    "title": review["title"],
                    "variety": review["variety"],
                    "winery": review["winery"],
                    "country": review["country"],
                    "province": review["province"],
                    "description": review["description"],
                    "points": review["points"],
                    "price": review.get("price"),
                    "taster_name": review.get("taster_name"),
                    "taster_twitter_handle": review.get("taster_twitter_handle"),
                })
            
            # Create embeddings for the entire batch
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch_contents
            )
            
            # Store each embedding with its metadata
            for index2, embedding_data in enumerate(response.data):
                metadata = batch_metadata[index2]
                embedding = embedding_data.embedding
                
                cursor.execute(
                    """INSERT INTO reviews 
                    (title, variety, winery, country, province, description, points, price, 
                        taster_name, taster_twitter_handle, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        metadata["title"],
                        metadata["variety"],
                        metadata["winery"],
                        metadata["country"],
                        metadata["province"],
                        metadata["description"],
                        metadata["points"],
                        metadata["price"],
                        metadata["taster_name"],
                        metadata["taster_twitter_handle"],
                        embedding
                    )
                )
                print(f"Stored embedding for: {metadata['title'][:50]}...")
            
            print(f"Processed batch {index1//batch_size + 1}/{(len(reviews) + batch_size - 1)//batch_size}")

        conn.commit()
        print("All embeddings stored successfully!")

    except Exception as e:
        print("Error generating embeddings:", e)

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    generate_embeddings()