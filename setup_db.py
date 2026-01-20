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
    # Extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id serial PRIMARY KEY,
            title text NOT NULL,
            variety text DEFAULT NULL,
            winery text DEFAULT NULL,
            country text DEFAULT NULL,
            province text DEFAULT NULL,
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
    print("Table created.")

    # Indexes
    print("Creating indexes...")

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_reviews_embedding_hnsw
        ON reviews USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    conn.commit()
    print("  - embedding (HNSW)")

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_reviews_taster_name_lower
        ON reviews (LOWER(taster_name));
    """)
    conn.commit()
    print("  - taster_name")

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_reviews_points
        ON reviews (points);
    """)
    conn.commit()
    print("  - points")

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_reviews_price
        ON reviews (price) WHERE price IS NOT NULL;
    """)
    conn.commit()
    print("  - price")

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_reviews_points_price
        ON reviews (points DESC NULLS LAST, price NULLS LAST);
    """)
    conn.commit()
    print("  - points_price")

    print("Database setup complete!")
except Exception as e:
    print("Error during setup:", e)
finally:
    cur.close()
    conn.close()