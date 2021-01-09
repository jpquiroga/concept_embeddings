import numpy as np
import spacy
from typing import Iterable, List, Text


class EmbeddingProcessor(object):

    def get_embedding(self, word: Text, sentence: Iterable[Text]) -> np.ndarray:
        """
        Get the embedding representation for a word within a sentence context.
        :param word: The word.
        :param sentence: The context sentence.
        :return: A `numpy` array containing the embedding.
        """
        raise NotImplemented("This method should be implemented by all child classes.")


class SpacyEmbeddingProcessor(EmbeddingProcessor):

    def __init__(self, language_model: Text, contextual_embedding: bool = False):
        """
        :param language_model: The Spacy language model to load. See https://spacy.io/models for details.
            Examples of models:
                - English:
                    - en_core_web_sm
                    - en_core_web_md
                    - en_core_web_lg
                    - en_vectors_web_lg (see https://spacy.io/models/en-starters#en_vectors_web_lg)
                    - en_trf_bertbaseuncased_lg (see https://spacy.io/models/en-starters#en_trf_bertbaseuncased_lg)
                    - en_trf_robertabase_lg (see https://spacy.io/models/en-starters#en_trf_robertabase_lg)
                    - en_trf_distilbertbaseuncased_lg (see https://spacy.io/models/en-starters#en_trf_distilbertbaseuncased_lg)
                    - en_trf_xlnetbasecased_lg (see https://spacy.io/models/en-starters#en_trf_xlnetbasecased_lg)
                - Spanish:
                    - es_core_news_sm
                    - es_core_news_md
                    - es_core_news_lg
                - French:
                    - fr_core_news_sm
                    - fr_core_news_md
                    - fr_core_news_lg
                - German:
                    - de_core_news_sm
                    - de_core_news_md
                    - de_core_news_lg
                    - de_trf_bertbasecased_lg (see https://spacy.io/models/de-starters)
                - Portuguese:
                    - pt_core_news_sm
                    - pt_core_news_md
                    - pt_core_news_lg
                - Italian:
                    - it_core_news_sm
                    - it_core_news_md
                    - it_core_news_lg
                - Multilanguage:
                    - xx_ent_wiki_sm
            :param contextual_embedding: Tells whether the embedding is contextual.

            Download models using `python -m spacy download <model_name>`.
        """
        self.nlp = spacy.load(language_model)
        self.language_model_name = language_model
        self.contextual = contextual_embedding

    def get_embedding(self, word: Text, sentence: Iterable[Text]) -> np.ndarray:
        tokens = list(self.nlp(sentence))
        tokens_index = {tok.text: i for i, tok in enumerate(tokens)}
        if self.contextual:
            return tokens[tokens_index[word]].vector
        else:
            word_vector = tokens[tokens_index[word]].vector
            # Calculate context vector as the mean of the context word s vectors.
            context_vector = np.zeros(len(word_vector))
            num_vectors = 0
            for tok in tokens:
                if tok.text != word:
                    context_vector += tok.vector
                    num_vectors += 1
            if num_vectors > 0:
                context_vector = context_vector / num_vectors
            # Return concatenated word and context vectors
            return np.concatenate([word_vector, context_vector])
