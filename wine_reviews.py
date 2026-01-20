import time
from concurrent.futures import ThreadPoolExecutor
import gradio as gr
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from database_helper import search_reviews, init_pool
from dotenv import load_dotenv
from mem0 import MemoryClient

load_dotenv()
init_pool()
llm_client = OpenAI()
memory_client = MemoryClient()
EMBEDDING_MODEL = 'text-embedding-3-small'
LLM_MODEL = 'gpt-4.1-nano'
USER_ID = 'wine-user-1'

class QueryClassification(BaseModel):
    type: str = Field(
        description=(
            "If the query mentions attributes like country, province, variety, or description, set type='semantic'. "
            "If it only specifies filters like price, points, or a tasters' name, set type='keyword'. "
            "If both are present, use 'semantic'."
        )
    )
    taster_name: Optional[str] = Field(default=None, description='The name of the taster (null if not mentioned).')
    min_points: Optional[int] = Field(default=None, description='The minimum points that the wine should have (null if not mentioned).')
    max_points: Optional[int] = Field(default=None, description='The maximum points that the wine should have (null if not mentioned).')
    min_price: Optional[float] = Field(default=None, description='The minimum price of the wine (null if not mentioned).')
    max_price: Optional[float] = Field(default=None, description='The maximum price of the wine (null if not mentioned).')

# Memory
def get_relevant_memories(query):
    filters = {'user_id': USER_ID}
    memories = memory_client.search(query, filters=filters, top_k=5)
    if not memories.get('results'):
        return ''
    memory_texts = [m['memory'] for m in memories['results']]
    return '\n'.join(f'- {text}' for text in memory_texts)

def store_interaction(query, response):
    messages = [
        {'role': 'user', 'content': query},
        {'role': 'assistant', 'content': response}
    ]
    memory_client.add(messages, user_id=USER_ID)

# Classification
def classify_query(query):
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
    print(response.output_parsed)
    return response.output_parsed

# Embedding
def embed_query_text(text):
    resp = llm_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text)
    return resp.data[0].embedding

# Summarization
def summarize_results_with_llm(query, results, memories=''):
    memory_context = f"User preferences from past interactions:\n{memories}\n\n" if memories else ''

    if not results:
        input_text = (
            f"{memory_context}"
            f"User query: {query}\n\n"
            "No search results were found. Reply in natural language saying no close matches were found "
            "and suggest trying different keywords."
        )
    else:
        lines = []
        for index, row in enumerate(results, start=1):
            price_str = f'${row["price"]}' if row["price"] else 'N/A'
            lines.append(
                f'{index}. {row["title"]} ({row["winery"]}) - '
                f'{row["country"]} | {row["points"]}pts, {price_str}'
            )
        results_text = '\n'.join(lines)
        input_text = (
            f"{memory_context}"
            f"User query: {query}\n\n"
            f"Results:\n{results_text}\n\n"
            "Summarize briefly: highlight top matches, note price/rating trends, recommend one to try first."
        )

    response = llm_client.responses.create(
        model=LLM_MODEL,
        input=input_text,
        max_output_tokens=450
    )
    return response.output_text.strip()

# Handling
def handle_search(user_query, top_k = 10, min_similarity = 0.05):
    if not user_query.strip():
        return 'Please enter a search query.'

    timings = {}
    total_start = time.perf_counter()

    def timed_memory_search():
        start = time.perf_counter()
        result = get_relevant_memories(user_query)
        timings['Memory search'] = time.perf_counter() - start
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

    if classification.type == 'semantic':
        start = time.perf_counter()
        embedding = embed_query_text(user_query)
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
        timings['DB retrieval'] = time.perf_counter() - start
    else:
        timings['Embedding'] = None

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
        timings['DB retrieval'] = time.perf_counter() - start

    start = time.perf_counter()
    response = summarize_results_with_llm(user_query, rows, memories)
    timings['Summarization'] = time.perf_counter() - start

    timings['Total'] = time.perf_counter() - total_start

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(store_interaction, user_query, response)
    executor.shutdown(wait=False)

    return format_response_with_timings(response, timings)

def format_duration(seconds):
    if seconds >= 1:
        return f'{seconds:.2f}s'
    return f'{seconds * 1000:.0f}ms'

def format_response_with_timings(response, timings):
    timing_rows = []
    for step, duration in timings.items():
        if step == 'Total':
            continue
        if duration is None:
            timing_rows.append(f'| {step} | N/A |')
        else:
            timing_rows.append(f'| {step} | {format_duration(duration)} |')
    timing_rows.append(f'| **Total** | **{format_duration(timings["Total"])}** |')

    timing_table = '\n'.join([
        '',
        '---',
        '**Timing Breakdown**',
        '| Step | Time |',
        '|------|------|',
        *timing_rows
    ])

    return response + timing_table

# Interface
with gr.Blocks(title="Wine Review Assistant") as demo:
    gr.Markdown("# Wine Review Assistant")
    gr.Markdown("Search for wines by describing what you're looking for - variety, region, taste profile, price range, or rating.")

    query_input = gr.Textbox(
        label="Search Query",
        placeholder="e.g., A fruity red wine from California under $30",
        lines=2
    )
    search_button = gr.Button("Search", variant="primary")
    results_output = gr.Markdown(label="Results")

    search_button.click(
        fn=handle_search,
        inputs=[query_input],
        outputs=results_output
    ).then(
        fn=lambda: gr.update(interactive=True),
        outputs=search_button
    )

    query_input.submit(
        fn=handle_search,
        inputs=[query_input],
        outputs=results_output
    ).then(
        fn=lambda: gr.update(interactive=True),
        outputs=search_button
    )

    # Disable button while loading
    search_button.click(
        fn=lambda: gr.update(interactive=False),
        outputs=search_button
    )

demo.launch()
