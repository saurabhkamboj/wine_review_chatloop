from dotenv import load_dotenv
from mem0 import MemoryClient

from config import USER_ID

load_dotenv()
memory_client = MemoryClient()


def get_relevant_memories(query: str) -> str:
    """Search for relevant memories based on the query."""
    filters = {'user_id': USER_ID}
    memories = memory_client.search(query, filters=filters, top_k=5)
    if not memories.get('results'):
        return ''
    memory_texts = [m['memory'] for m in memories['results']]
    return '\n'.join(f'- {text}' for text in memory_texts)


def store_interaction(query: str, response: str, image_description: str | None = None):
    """Store interaction in memory. Only stores text, not image URLs."""
    text_content = query
    if image_description:
        text_content = f"{query}\n\n[Analyzed image showed: {image_description}]"

    messages = [
        {'role': 'user', 'content': text_content},
        {'role': 'assistant', 'content': response}
    ]
    memory_client.add(messages, user_id=USER_ID)


def get_all_memories() -> list[str]:
    """Get all stored memories for the user."""
    try:
        filters = {'user_id': USER_ID}
        memories = memory_client.get_all(filters=filters)
        if not memories.get('results'):
            return []
        return [m['memory'] for m in memories['results']]
    except Exception:
        return []
