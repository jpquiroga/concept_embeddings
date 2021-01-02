from elasticsearch import Elasticsearch
import json
from typing import List, Text

class ElasticSearchQueryManager(object):

    MAX_SENTENCES_TO_PROCESS = 1000
    SENTENCE_MIN_LEN = 5

    def get_sentences_by_word(self, word: str, e_s: Elasticsearch, es_index: Text = None,
                            max_sentences_to_process: int = MAX_SENTENCES_TO_PROCESS,
                            sentence_min_len: int = SENTENCE_MIN_LEN) -> List[List[str]]:
        """
        Get the sentences in the database containing a given word.

        :param word: Word to look for in ElasticSearch.
        :param e_s: ElasticSearch client.
        :param es_index: The name of the ElasticSearch index to look into.
        :param max_sentences_to_process: Maximum number of sentences to process.
        :param sentence_min_len: The minimum length of the sentences to be returned. Sentences with length smaller than
            `sentence_min_len` will not be returned.
        :return: A list with the found sentences. Sentences are returned as a lists of normalized words.
        """
        q = {
          "query": {
            "match": {
              "text": word
            }
          }
        }
        e_s_res = e_s.search(body=json.dumps(q), index=es_index, size=max_sentences_to_process)
        e_s_hits = e_s_res["hits"]["hits"]
        candidate_res = [r["_source"]["text"].split(" ") for r in e_s_hits]
        res = [s for s in candidate_res if word in s and len(s) >= sentence_min_len]
        return res
