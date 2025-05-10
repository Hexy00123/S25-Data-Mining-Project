import re

def simplify_category_list(category_string):
    """
    Takes a space-separated string of arXiv categories (e.g., "math.AP cs.AI hep-th")
    and returns a sorted list of unique simplified categories (e.g., ['cs', 'hep-th', 'math']).
    """
    if not category_string or not isinstance(category_string, str):
        return []
    categories = category_string.split()
    simplified_categories = set()
    for cat in categories:
        if '.' in cat:
            simplified_categories.add(cat.split('.')[0])
        else:
            simplified_categories.add(cat) # Keep categories without a dot as is
    return sorted(list(simplified_categories))

def preprocess_text(text):
    """
    Cleans text data: lowercase, filter short/non-alpha words.
    Designed for title/abstract concatenation.
    """
    if not isinstance(text, str): # Handle potential non-string data
        return ""
    text = text.lower()
    # Keep only alphabetic words longer than 3 chars
    words = [word for word in text.split() if len(word) > 3 and word.isalpha()]
    return ' '.join(words)

def preprocess_categories_and_text_batch(batch):
    """
    Applies category simplification and text preprocessing to a dataset batch.
    Adds 'categories_simplified_list' and 'text_processed' columns.
    Designed for use with datasets.map().
    """
    # Simplify Categories
    batch['categories_simplified_list'] = [simplify_category_list(cats) for cats in batch['categories']]

    # Preprocess Text (Combine title and abstract)
    texts_to_process = [(title if title else "") + ' ' + (abstract if abstract else "")
                        for title, abstract in zip(batch['title'], batch['abstract'])]
    batch['text_processed'] = [preprocess_text(text) for text in texts_to_process]

    return batch
