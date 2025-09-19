import json
import gradio as gr
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional
from database_helper import search_reviews
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
EMBEDDING_MODEL = 'text-embedding-3-small'
LLM_MODEL = 'gpt-4o-mini'

# Pydantic schema for classification
from typing import Optional
from pydantic import BaseModel, Field

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

def classify_query(query):
    response = client.responses.parse(
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
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text)
    return resp.data[0].embedding

# Summarization
def summarize_results_with_llm(query, results):
    if not results:
        input_text = (
            f"User query: {query}\n\n"
            "No search results were found. Reply in natural language saying no close matches were found "
            "and suggest trying different keywords."
        )
    else:
        lines = []
        for index, row in enumerate(results, start=1):
            similarity = f'{row["similarity"]:.3f}' if row["similarity"] is not None else "N/A"
            lines.append(
                f'{index}. {row["title"]} — {row["winery"]} '
                f'({row["country"]}, {row["province"]}) | '
                f'Points: {row["points"]}, Price: {row["price"]} | '
                f'Taster: {row["taster_name"] or "N/A"} '
                f'[similarity={similarity}]'
            )
        results_text = '\n'.join(lines)
        input_text = (
            f"User query: {query}\n\n"
            f"Found {len(results)} similar wine reviews. Provide a short human-friendly summary that:\n"
            "1) highlights the most relevant wines and why they match the query,\n"
            "2) mentions regions, points, and price trends if visible,\n"
            "3) suggests which wine the user should explore first based on intent.\n\n"
            f"Results:\n{results_text}\n\n"
            "Respond with a short descriptive paragraph followed by a concise numbered recommendation list."
        )

    response = client.responses.create(
        model=LLM_MODEL,
        input=input_text,
        max_output_tokens=700
    )
    return response.output_text.strip()

def handle_search(user_query, top_k = 10, min_similarity = 0.05):
    if not user_query.strip():
        return 'Please enter a search query.'

    classification = classify_query(user_query)

    if classification.type == 'semantic':
        embedding = embed_query_text(user_query)
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
    else:
        rows = search_reviews(
            query_embedding=None,
            top_k=top_k,
            taster_name=classification.taster_name,
            min_points=classification.min_points,
            max_points=classification.max_points,
            min_price=classification.min_price,
            max_price=classification.max_price
        )

    return summarize_results_with_llm(user_query, rows)

# Create the Gradio interface
with gr.Blocks(title="Wine review assistant") as demo:
    gr.Markdown("# 🍷 Wine review assistant")
    gr.Markdown("Search for relevant wine reviews using natural language queries.")
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Search Query",
                placeholder="Enter your search query here...",
                lines=2
            )
            search_button = gr.Button("Search", variant="primary")
        
        with gr.Column():
            results_output = gr.Markdown(label="Results")

    search_button.click(
        fn=handle_search,
        inputs=[query_input],
        outputs=results_output
    )

    query_input.submit(
        fn=handle_search,
        inputs=[query_input],
        outputs=results_output
    )

demo.launch()
