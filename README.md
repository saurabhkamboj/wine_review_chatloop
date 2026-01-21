# Wine Review Chatbot

A CLI chatbot for exploring 130k+ wine reviews.

## Setup

1. Install dependencies with Poetry:

    ```bash
    poetry install
    ```

2. Activate the virtual environment:

    ```bash
    eval $(poetry env activate)
    ```

3. Set up your `.env` file:

    ```bash
    OPENAI_API_KEY=your_api_key_here
    MEM0_API_KEY=your_mem0_api_key_here
    ```

4. Update the database connection settings in `setup_db.py` and `database_helper.py` with your PostgreSQL credentials (host, port, user, database name).

5. Create and populate the PostgreSQL database:

    ```bash
    python setup_db.py
    python load_embeddings.py
    ```

6. Run the application:

    ```bash
    python cli/main.py
    ```

## Commands

- `/quit`, `/exit` - Exit the application
- `/clear` - Clear conversation history
- `/memories` - Show stored preferences
- `/help` - Show help message
