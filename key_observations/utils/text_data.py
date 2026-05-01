import numpy as np
from datasets import load_dataset
from nltk.tokenize import word_tokenize

try:
    word_tokenize('Arbitrary sentence.')
except:
    import nltk
    nltk.download('punkt_tab')



def get_random_long_text(dataset_name: str,
                         min_word_count: int = 500,
                         max_word_count: int = 700,
                         split: str = 'train',
                         random_seed: int = 0) -> dict:
    if dataset_name == 'wikipedia':
        dataset = load_dataset("wikitext", "wikitext-103-v1")
        key = 'text'
    elif dataset_name == 'pubmed':
        dataset = load_dataset("pubmed_qa", "pqa_labeled")
        key = 'long_answer'
    elif dataset_name == 'imdb':
        dataset = load_dataset("imdb")
        key = 'text'
    elif dataset_name == 'squad':
        dataset = load_dataset("squad")
        key = 'context'

    text = ''
    rng = np.random.default_rng(seed=random_seed)
    idx = rng.integers(0, int(len(dataset['train']) * 0.95)).item()
    while len(word_tokenize(text)) < min_word_count:
        text += dataset[split][idx][key]
        idx += 1
        if len(word_tokenize(text)) > max_word_count:
            break
    return text
