import json
from openai import OpenAI
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

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
        
        # Process reviews in batches
        for index1 in range(0, len(reviews), batch_size):
            batch = reviews[index1:index1 + batch_size]
            
            # Prepare batch data
            batch_contents = []
            batch_metadata = []
            
            for review in batch:
                content = f"""
                    Title: {review['title']}
                    Country: {review.get('country')}
                    Province: {review.get('province')}
                    Variety: {review.get('variety')}
                    Description: {review['description']}
                    """
                batch_contents.append(content)
                batch_metadata.append({
                    "title": review["title"],
                    "variety": review.get("variety"),
                    "winery": review.get("winery"),
                    "country": review.get("country"),
                    "province": review.get("province"),
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
            
            # Prepare rows for bulk insert
            rows_to_insert = []
            for index2, embedding_data in enumerate(response.data):
                metadata = batch_metadata[index2]
                embedding = embedding_data.embedding
                
                rows_to_insert.append((
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
                ))
            
            # Bulk insert with execute_values
            execute_values(cursor, """
                INSERT INTO reviews 
                (title, variety, winery, country, province, description, points, price, 
                 taster_name, taster_twitter_handle, embedding)
                VALUES %s
                ON CONFLICT DO NOTHING
            """, rows_to_insert)
            
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
