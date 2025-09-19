import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    user="saurabhkamboj",
    database="wine_reviews"
)

cur = conn.cursor()

try:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id serial PRIMARY KEY,
            title text NOT NULL,
            variety text NOT NULL,
            winery text NOT NULL,
            country text NOT NULL,
            province text NOT NULL,
            description text NOT NULL,
            points integer NOT NULL CHECK (points BETWEEN 0 AND 100),
            price numeric CHECK (price IS NULL OR price >= 0),
            taster_name text DEFAULT NULL,
            taster_twitter_handle text DEFAULT NULL,
            embedding vector(1536),
            UNIQUE (title, winery, description, taster_name)
        );
    """)
    conn.commit()
    print("Database setup complete!")
except Exception as e:
    print("Error during setup:", e)
finally:
    cur.close()
    conn.close()