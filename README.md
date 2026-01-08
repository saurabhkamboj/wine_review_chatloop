# Wine-review chatbot

A semantic search chatbot for exploring wine reviews.

## Features

- Search 130k wine reviews using natural language.
- Semantic and keyword-based filtering using structured outputs.

## Setup

1. Install dependencies with Poetry:

    ```bash
    poetry install
    ```

2. Set up your `.env` file with your OpenAI API key:

    ```bash
    OPENAI_API_KEY=your_api_key_here
    ```

3. Update the database connection settings in `setup_db.py` and `database_helper.py` with your PostgreSQL credentials (host, port, user, database name).

4. Create and populate the PostgreSQL database:

    ```bash
    python setup_db.py
    python load_embeddings.py
    ```

5. Run the application:

    ```bash
    python wine_reviews.py
    ```
