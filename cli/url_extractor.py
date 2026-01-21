import re

# Pattern to match common image URLs
URL_PATTERN = r'https?://[^\s<>"\']+(?:\.(?:jpg|jpeg|png|gif|webp|bmp))[^\s<>"\']*'


def extract_image_urls(text: str) -> tuple[str, list[str]]:
    """
    Extract image URLs from text.

    Returns:
        tuple: (cleaned_text, list_of_image_urls)
    """
    image_urls = re.findall(URL_PATTERN, text, re.IGNORECASE)
    cleaned_text = re.sub(URL_PATTERN, '', text, flags=re.IGNORECASE).strip()
    # Clean up extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text, image_urls
