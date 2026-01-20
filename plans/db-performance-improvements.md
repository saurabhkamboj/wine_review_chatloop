# Plan: Fix Slow Database Retrieval (5-10 seconds)

## Problem

DB retrieval currently takes 5-10 seconds due to:

1. **No vector index** on `embedding` column — pgvector does sequential scan on 130k vectors
2. **No B-tree indexes** on filtered columns (`taster_name`, `points`, `price`)
3. **COALESCE anti-pattern** in WHERE clause prevents index usage
4. **New connection per query** (~50-100ms overhead)

## Changes

### 1. Add HNSW Vector Index (Highest Impact)

Update [setup_db.py](setup_db.py) to add:

```sql
CREATE INDEX IF NOT EXISTS idx_reviews_embedding_hnsw
ON reviews USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

- HNSW gives ~10-50x speedup for vector similarity search
- Build time: ~2-5 minutes for 130k rows

### 2. Add B-tree Indexes

```sql
-- For taster_name filter (expression index to match LOWER() in query)
CREATE INDEX IF NOT EXISTS idx_reviews_taster_name_lower
ON reviews (LOWER(taster_name));

-- For points/price range filters
CREATE INDEX IF NOT EXISTS idx_reviews_points ON reviews (points);
CREATE INDEX IF NOT EXISTS idx_reviews_price ON reviews (price) WHERE price IS NOT NULL;

-- For keyword search ORDER BY
CREATE INDEX IF NOT EXISTS idx_reviews_points_price
ON reviews (points DESC NULLS LAST, price NULLS LAST);
```

### 3. Add Connection Pooling

Update [database_helper.py](database_helper.py):

```python
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager

_pool = None

def init_pool():
    global _pool
    _pool = ThreadedConnectionPool(
        minconn=2, maxconn=20,
        host="localhost", port=5432,
        user="saurabhkamboj", database="wine_reviews"
    )

@contextmanager
def get_connection():
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)
```

Update [wine_reviews.py](wine_reviews.py) to call `init_pool()` at startup.

### 4. Fix COALESCE Anti-pattern

Refactor `search_reviews()` in [database_helper.py](database_helper.py) to build WHERE dynamically:

```python
conditions = []
params = []

if taster_name is not None:
    conditions.append("LOWER(taster_name) = LOWER(%s)")
    params.append(taster_name)

if min_points is not None:
    conditions.append("points >= %s")
    params.append(min_points)
# ... etc

where_clause = " AND ".join(conditions) if conditions else "TRUE"
```

## Files to Modify

| File | Changes |
| ---- | ------- |
| [setup_db.py](setup_db.py) | Add 5 CREATE INDEX statements |
| [database_helper.py](database_helper.py) | Add connection pooling + refactor query building |
| [wine_reviews.py](wine_reviews.py) | Call `init_pool()` at startup |

## Expected Performance

| Query Type | Before | After |
| ---------- | ------ | ----- |
| Semantic search | 5-10s | ~0.1-0.3s |
| Keyword search | 1-2s | ~0.01-0.05s |

## Verification

1. Run `python setup_db.py` to create indexes (takes 2-5 min for HNSW)
2. Start app with `python wine_reviews.py`
3. Submit query and check timing breakdown — DB retrieval should be <0.5s
4. Run `EXPLAIN ANALYZE` on a sample query to confirm index usage
