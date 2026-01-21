from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from config import COLORS

console = Console()


def print_welcome():
    """Print welcome message with instructions."""
    welcome_text = Text()
    welcome_text.append("Wine Review Assistant\n", style="bold")
    welcome_text.append("\nSearch for wines by describing what you're looking for.\n")
    welcome_text.append("Include image URLs to analyze wine labels.\n\n")
    welcome_text.append("Commands:\n", style="bold")
    welcome_text.append("  /quit, /exit  - Exit the application\n")
    welcome_text.append("  /clear        - Clear conversation history\n")
    welcome_text.append("  /memories     - Show stored preferences\n")
    welcome_text.append("  /help         - Show this message\n")

    console.print(Panel(welcome_text, border_style="dim"))
    console.print()


def print_user(message: str):
    """Print user input with styling."""
    console.print(f"[{COLORS['user']}]You:[/{COLORS['user']}] {message}")


def print_assistant_start():
    """Print assistant label before streaming."""
    console.print(f"[{COLORS['assistant']}]Assistant:[/{COLORS['assistant']}] ", end="")


def print_error(message: str):
    """Print error message."""
    console.print(f"[{COLORS['error']}]Error: {message}[/{COLORS['error']}]")


def print_timing(timings: dict):
    """Print timing information as a formatted string."""
    parts = []
    order = ['Memory', 'Classification', 'Image', 'Embedding', 'DB']

    for key in order:
        if key in timings:
            parts.append(f"{key}: {format_duration(timings[key])}")

    if 'Total' in timings:
        parts.append(f"Total: {format_duration(timings['Total'])}")

    timing_str = " | ".join(parts)
    console.print(f"[{COLORS['timing']}]{timing_str}[/{COLORS['timing']}]")


def format_duration(seconds: float) -> str:
    """Format duration in appropriate units."""
    if seconds >= 1:
        return f'{seconds:.2f}s'
    return f'{seconds * 1000:.0f}ms'
