import json
import string
import random


def generate_random_string(length=10):
    """Generate a random string of letters and digits."""
    random_string = "".join(
        random.choices(string.ascii_letters + string.digits, k=length)
    )
    return random_string


def load_json(path):
    """Load a json file into a list of dictionaries

    Args:
        path (str): path to jsonl file

    Returns:
        List[dict]: list of dictionaries
    """
    all_texts = []
    with open(path, "r") as f:
        data = json.load(f)
        for author in data:
            current_texts = []
            if "quote" in author:
                quotes = [quote.strip("“”") for quote in author["quote"]]
                current_texts.extend(quotes)
            current_texts.append(author["description"].strip("Quotes to Scrape Login"))
            all_texts.append(" ".join(current_texts))
    return all_texts
