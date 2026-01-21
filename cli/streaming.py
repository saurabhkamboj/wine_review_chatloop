from dotenv import load_dotenv
from openai import OpenAI
from rich.live import Live
from rich.text import Text

from config import LLM_MODEL, COLORS
from cli.console import console

load_dotenv()
llm_client = OpenAI()


def stream_response(prompt: str) -> str:
    """
    Stream LLM response to console using Rich Live.

    Returns the full response text.
    """
    full_response = ""

    stream = llm_client.responses.create(
        model=LLM_MODEL,
        input=prompt,
        stream=True
    )

    with Live(Text("", style=COLORS['assistant']), console=console, refresh_per_second=15) as live:
        for event in stream:
            if event.type == "response.output_text.delta":
                full_response += event.delta
                live.update(Text(full_response, style=COLORS['assistant']))

    return full_response


def build_prompt(
    query: str,
    results_text: str,
    memories: str = '',
    image_description: str | None = None,
    conversation_history: str = ''
) -> str:
    """Build the LLM prompt with all context."""
    context_sections = []

    if memories:
        context_sections.append(
            f'## Memory Context\nUser preferences from past interactions:\n{memories}'
        )

    if conversation_history:
        context_sections.append(
            f'## Conversation History\n{conversation_history}'
        )

    if image_description:
        context_sections.append(
            f'## Image Context\nUser provided an image showing: {image_description}'
        )

    context_text = '\n\n'.join(context_sections) + '\n\n' if context_sections else ''

    if not results_text:
        return (
            f"{context_text}"
            f"## User Query\n{query}\n\n"
            "No search results were found. Reply in natural language saying no close matches were found "
            "and suggest trying different keywords."
        )

    return (
        f"{context_text}"
        f"## User Query\n{query}\n\n"
        f"## Search Results\n{results_text}\n\n"
        "Summarize the results based on the user query and memory context. Include relevant details like variety, location, reviewer/taster name, price, and points. If memory indicates user preferences (e.g., wanting taster names), ensure those are included in your response."
    )
