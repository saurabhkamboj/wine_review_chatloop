import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

from config import EMBEDDING_MODEL, LLM_MODEL, VISION_MODEL
from core.models import QueryClassification
from core.memory import get_relevant_memories
from database_helper import search_reviews

load_dotenv()
llm_client = OpenAI()


@dataclass
class SearchResult:
    """Container for search results and metadata."""
    results: list[dict]
    memories: str
    image_description: str | None
    classification: QueryClassification
    timings: dict = field(default_factory=dict)


def classify_query(query: str) -> QueryClassification:
    """Classify the query to determine search type and extract filters."""
    response = llm_client.responses.parse(
        model=LLM_MODEL,
        input=[
            {
                'role': 'system',
                'content': 'Extract information required to classify the query'
            },
            {'role': 'user', 'content': query}
        ],
        text_format=QueryClassification
    )
    return response.output_parsed


def describe_image(image_url: str) -> str:
    """Use vision LLM to describe a wine image."""
    response = llm_client.responses.create(
        model=VISION_MODEL,
        input=[
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'input_text',
                        'text': 'Describe this wine image briefly. Focus on: wine type, color, label details, region/origin if visible. Keep it concise.'
                    },
                    {'type': 'input_image', 'image_url': image_url}
                ]
            }
        ],
        max_output_tokens=150
    )
    return response.output_text.strip()


def embed_query(text: str) -> list[float]:
    """Generate embedding for the query text."""
    resp = llm_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def prepare_search(
    user_query: str,
    image_urls: list[str] | None = None,
    top_k: int = 10,
    min_similarity: float = 0.05
) -> SearchResult:
    """
    Prepare and execute search with parallel memory search and classification.

    Returns SearchResult with results, memories, and timing info.
    """
    timings = {}
    total_start = time.perf_counter()
    image_description = None
    memories = ''

    # Get image description if URL provided
    if image_urls:
        start = time.perf_counter()
        image_description = describe_image(image_urls[0])
        timings['Image'] = time.perf_counter() - start

    # Build memory search query
    memory_query = f'{user_query} {image_description}' if image_description else user_query

    # Memory search and classification in parallel
    def timed_memory_search():
        start = time.perf_counter()
        result = get_relevant_memories(memory_query)
        timings['Memory'] = time.perf_counter() - start
        return result

    def timed_classify():
        start = time.perf_counter()
        result = classify_query(user_query)
        timings['Classification'] = time.perf_counter() - start
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        memory_future = executor.submit(timed_memory_search)
        classify_future = executor.submit(timed_classify)
        memories = memory_future.result()
        classification = classify_future.result()

    # Build search text
    search_components = [user_query]
    if image_description:
        search_components.append(image_description)
    if memories:
        search_components.append(memories)
    search_text = ' '.join(search_components)

    # Embedding and DB search
    if classification.type == 'semantic' or image_description or memories:
        start = time.perf_counter()
        embedding = embed_query(search_text)
        timings['Embedding'] = time.perf_counter() - start

        start = time.perf_counter()
        rows = search_reviews(
            query_embedding=embedding,
            top_k=top_k,
            min_similarity=min_similarity,
            taster_name=classification.taster_name,
            min_points=classification.min_points,
            max_points=classification.max_points,
            min_price=classification.min_price,
            max_price=classification.max_price
        )
        timings['DB'] = time.perf_counter() - start
    else:
        start = time.perf_counter()
        rows = search_reviews(
            query_embedding=None,
            top_k=top_k,
            taster_name=classification.taster_name,
            min_points=classification.min_points,
            max_points=classification.max_points,
            min_price=classification.min_price,
            max_price=classification.max_price
        )
        timings['DB'] = time.perf_counter() - start

    timings['Total'] = time.perf_counter() - total_start

    return SearchResult(
        results=rows,
        memories=memories,
        image_description=image_description,
        classification=classification,
        timings=timings
    )


def format_results_for_prompt(results: list[dict]) -> str:
    """Format search results for inclusion in LLM prompt."""
    if not results:
        return ''

    lines = []
    for index, row in enumerate(results, start=1):
        price_str = f'${row["price"]}' if row["price"] else 'N/A'
        location = ', '.join(filter(None, [row.get("province"), row.get("country")]))
        reviewer = row.get("taster_name") or 'Unknown'
        variety = row.get("variety") or 'N/A'
        description = row.get("description") or ''

        lines.append(
            f'{index}. **{row["title"]}** ({row["winery"]})\n'
            f'   Variety: {variety} | Location: {location}\n'
            f'   Points: {row["points"]} | Price: {price_str} | Reviewer: {reviewer}\n'
            f'   Description: {description}'
        )
    return '\n\n'.join(lines)
