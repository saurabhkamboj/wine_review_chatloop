from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

from config import MAX_HISTORY_MESSAGES
from cli.console import (
    console,
    print_welcome,
    print_user,
    print_assistant_start,
    print_error,
    print_timing,
)
from cli.url_extractor import extract_image_urls
from cli.streaming import stream_response, build_prompt
from core.search import prepare_search, format_results_for_prompt
from core.memory import store_interaction, get_all_memories
from database_helper import init_pool


class ConversationHistory:
    """Manages conversation history for the current session."""

    def __init__(self, max_messages: int = MAX_HISTORY_MESSAGES):
        self.exchanges: list[tuple[str, str]] = []
        self.max_messages = max_messages

    def add_exchange(self, user_msg: str, assistant_msg: str):
        """Add a user-assistant exchange to history."""
        self.exchanges.append((user_msg, assistant_msg))
        # Keep only the last max_messages exchanges
        if len(self.exchanges) > self.max_messages:
            self.exchanges = self.exchanges[-self.max_messages:]

    def get_context_string(self) -> str:
        """Format conversation history for LLM prompt."""
        if not self.exchanges:
            return ''

        lines = []
        for user_msg, assistant_msg in self.exchanges:
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {assistant_msg}")
        return '\n'.join(lines)

    def clear(self):
        """Clear all conversation history."""
        self.exchanges = []


class WineChatbot:
    """Main chatbot class handling the CLI chatloop."""

    def __init__(self):
        self.history = ConversationHistory()

    def handle_command(self, command: str) -> bool:
        """
        Handle special commands.

        Returns True if command was handled (should continue loop).
        Returns False if should exit.
        """
        cmd = command.lower().strip()

        if cmd in ('/quit', '/exit'):
            console.print("\nGoodbye!")
            return False

        if cmd == '/clear':
            self.history.clear()
            console.print("[dim]Conversation history cleared.[/dim]")
            return True

        if cmd == '/memories':
            memories = get_all_memories()
            if memories:
                console.print("[bold]Stored Preferences:[/bold]")
                for memory in memories:
                    console.print(f"  - {memory}")
            else:
                console.print("[dim]No stored memories found.[/dim]")
            return True

        if cmd == '/help':
            print_welcome()
            return True

        return True  # Unknown command, treat as query

    def process_query(self, user_input: str):
        """Process a user query and generate response."""
        # Extract image URLs from input
        cleaned_query, image_urls = extract_image_urls(user_input)

        if not cleaned_query.strip():
            print_error("Please enter a search query.")
            return

        # Run search with parallel operations
        try:
            search_result = prepare_search(
                user_query=cleaned_query,
                image_urls=image_urls if image_urls else None
            )
        except Exception as e:
            print_error(f"Search failed: {e}")
            return

        # Format results for prompt
        results_text = format_results_for_prompt(search_result.results)

        # Build prompt with all context
        prompt = build_prompt(
            query=cleaned_query,
            results_text=results_text,
            memories=search_result.memories,
            image_description=search_result.image_description,
            conversation_history=self.history.get_context_string()
        )

        # Stream response
        print_assistant_start()
        try:
            response = stream_response(prompt)
        except Exception as e:
            console.print()  # Newline after assistant label
            print_error(f"Failed to generate response: {e}")
            return

        console.print()  # Newline after streamed response

        # Print timing
        print_timing(search_result.timings)
        console.print()

        # Update conversation history
        self.history.add_exchange(cleaned_query, response)

        # Store in Mem0 (background)
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(
            store_interaction,
            cleaned_query,
            response,
            search_result.image_description
        )
        executor.shutdown(wait=False)

    def run(self):
        """Main chatloop."""
        print_welcome()

        while True:
            try:
                user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Check for commands
            if user_input.startswith('/'):
                if not self.handle_command(user_input):
                    break
                # If it was a recognized command, continue
                if user_input.lower() in ('/quit', '/exit', '/clear', '/memories', '/help'):
                    continue

            # Process as query
            self.process_query(user_input)


def main():
    """Entry point for the CLI application."""
    load_dotenv()
    init_pool()

    chatbot = WineChatbot()
    chatbot.run()


if __name__ == '__main__':
    main()
