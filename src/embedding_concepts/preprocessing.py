import datasets
import spacy
from elasticsearch import Elasticsearch
from datetime import datetime
from tqdm import tqdm
from typing import Text
#from .constants import WIKITEXT_DATASET, ES_QUERY_SIZE, ES_INDEX


# Identify sentences and send them to ElasticSearch to be indexed. Use Spacy for sentence tokenization.

class Normalizer(object):
    PUNCTUATION_CHARS = [".", ",", ":", ";", "?", "!", "¿", '"', "'"]

    @staticmethod
    def get_normalized_sentence(s: spacy.tokens.span.Span) -> str:
        """
        Transform a spacy Span object into a normalized string sentence:
        - Lower case.
        - Without punctuation chars.
        """
        res = [t.text.lower() for t in s if t.text not in Normalizer.PUNCTUATION_CHARS]
        return " ".join(res).strip()


def send_to_elasticsearch(s: Text, model: "spacy.language.Language", e_s: Elasticsearch, e_s_index: str):
    """
    Send to Elastic Search the sentences contained in a given text.
    The text is split into sentences, normalized an then sent to ElasticSearch (one document per sentence).

    :param s: String to process.
    :param model: Spacy model to be used for sentence chunking and tokenization.
    :param e_s: ElasticSearch object to invoke indexing.
        Use, for instance:

        .. code-block:: python

            es = Elasticsearch(
                hosts=["localhost:9200"],
            #    sniff_on_start=True,
            #    sniff_on_connection_fail=True,
            #    sniffer_timeout=60
            )

    :param e_s_index: The name of the ElasticSearch index to be used.
    """
    doc = model(s)
    s_sentences = [Normalizer.get_normalized_sentence(sent) for sent in doc.sents]
    orig_s_sentences = [str(sent) for sent in doc.sents]
    for i in range(len(s_sentences)):
        doc = {
            'original_text': orig_s_sentences[i],
            'text': s_sentences[i],
            'timestamp': datetime.now(),
        }
        res = e_s.index(index=e_s_index, body=doc)


def index_sentences(wikitext_dataset: datasets.arrow_dataset.Dataset, nlp: "spacy.language.Language",
                    e_s: Elasticsearch, es_index: str):
    """
    :param wikitext_dataset: Dataset. Example:

    .. code-block:: pyhton

        wikitext_dataset = load_dataset('wikitext', WIKITEXT_DATASET)
        train_wikitext_dataset = wikitext_dataset["train"]

    :param nlp: Spacy model. For instance, use `nlp = spacy.load("en_core_web_sm")` if language is English.
    :param e_s: ElasticSearch object to invoke indexing.
    :param es_index: The name of the ElasticSearch index to be used.
    """
    for d in tqdm(wikitext_dataset):
        send_to_elasticsearch(d["text"], nlp, e_s, es_index)
