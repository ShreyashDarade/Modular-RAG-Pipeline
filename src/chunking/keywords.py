from __future__ import annotations

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

# Python's \w does not match Devanagari vowel signs and viramas, which would shred Hindi/Marathi
# words into fragments; include the whole block explicitly.
_TOKEN_PATTERN = r"(?u)[\wऀ-ॿ]{2,}"


class KeywordExtractor:
    """Top TF-IDF terms (unigrams and bigrams) per text, computed over the batch given."""

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def extract(self, texts: Sequence[str]) -> list[list[str]]:
        if not texts:
            return []
        vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=_TOKEN_PATTERN,
            max_features=self.top_k * 2,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0 if len(texts) == 1 else 0.95,
        )
        try:
            matrix = vectorizer.fit_transform(texts).tocsr()
        except ValueError as exc:
            # A batch made only of stop words / punctuation legitimately has no vocabulary.
            if "empty vocabulary" in str(exc) or "no terms remain" in str(exc):
                return [[] for _ in texts]
            raise
        names = vectorizer.get_feature_names_out()
        keywords: list[list[str]] = []
        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row], matrix.indptr[row + 1]
            columns, weights = matrix.indices[start:end], matrix.data[start:end]
            order = weights.argsort()[::-1][: self.top_k]
            keywords.append([str(names[columns[i]]) for i in order])
        return keywords
