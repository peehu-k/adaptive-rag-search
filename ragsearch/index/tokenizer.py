"""Text analysis: turn a string into a list of index/query terms.

The analyzer is deliberately small and fully configurable. Which knobs are
set (stopwords, stemming, minimum length) is exactly the kind of choice the
search loop is meant to mutate later, so everything lives on one frozen
config object that a pipeline config can carry around.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# A short, conventional English stop list. Not exhaustive by design -- an
# aggressive list hurts recall on short scientific claims.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be by for from has have he in is it its of on or that
    the to was were will with this these those which who whom whose what when
    where why how not no nor but if then than so such can could would should
    may might must do does did done being been their they them her his our us
    we you your i me my
    """.split()
)


def light_stem(term: str) -> str:
    """Cheap, deterministic suffix stripping. Not a full Porter stemmer.

    Handles the high-frequency inflections (plurals, ``-ing`` / ``-ed``)
    that otherwise split a concept across several postings lists.
    """
    if len(term) <= 3:
        return term
    for suffix in ("ational", "ization", "iveness", "fulness", "ousness"):
        if term.endswith(suffix) and len(term) > len(suffix) + 2:
            return term[: -len(suffix)]
    if term.endswith("ies") and len(term) > 4:
        return term[:-3] + "y"
    if term.endswith("sses"):
        return term[:-2]
    if term.endswith("ing") and len(term) > 5:
        stem = term[:-3]
        return stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem
    if term.endswith("ed") and len(term) > 4:
        stem = term[:-2]
        return stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else stem
    if term.endswith("s") and not term.endswith(("ss", "us", "is")):
        return term[:-1]
    return term


@dataclass(frozen=True)
class Analyzer:
    lowercase: bool = True
    remove_stopwords: bool = True
    stem: bool = False
    min_token_len: int = 2

    def __call__(self, text: str) -> list[str]:
        raw = _TOKEN_RE.findall(text)
        out: list[str] = []
        for tok in raw:
            if self.lowercase:
                tok = tok.lower()
            if len(tok) < self.min_token_len:
                continue
            if self.remove_stopwords and tok.lower() in STOPWORDS:
                continue
            if self.stem:
                tok = light_stem(tok)
                if not tok:
                    continue
            out.append(tok)
        return out


DEFAULT_ANALYZER = Analyzer()
