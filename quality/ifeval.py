"""IFEval verifiable-instruction checks, ported to the stdlib.

Ported from google-research/instruction_following_eval (instructions.py and
instructions_util.py, Apache License 2.0, Copyright The Google Research
Authors). Each check below mirrors that file's check_following logic.

Deviations, all because nltk/langdetect are not stdlib:
  * count_sentences uses the upstream regex splitter (split_into_sentences)
    instead of the punkt model.
  * capital_word_frequency tokenizes with a regex instead of nltk.word_tokenize.
  * english_capital / english_lowercase test "English" as >= 90% of letters
    being ASCII, instead of langdetect.
  * language:response_language is not implemented; tier1.py never samples
    items that use it (31 of 541).
Scores are used only for paired A/B comparisons, where a shared bias cancels.
"""
from __future__ import annotations

import collections
import json
import re

UNSUPPORTED = {"language:response_language"}

_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def split_into_sentences(text: str) -> list[str]:
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(_MULTIPLE_DOTS, lambda m: "<prd>" * len(m.group(0)) + "<stop>", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>").replace("?", "?<stop>").replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = [s.strip() for s in text.split("<stop>")]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def count_sentences(text: str) -> int:
    return len([s for s in split_into_sentences(text) if s])


def _compare(actual: int, relation: str, threshold: int) -> bool:
    if relation == "less than":
        return actual < threshold
    if relation == "at least":
        return actual >= threshold
    raise ValueError(f"unknown relation {relation!r}")


def _is_english(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and sum(ch.isascii() for ch in letters) >= 0.9 * len(letters)


def _paragraph_first_word(value: str, num_paragraphs: int, nth: int, first_word: str) -> bool:
    paragraphs = re.split(r"\n\n", value)
    n = len(paragraphs) - sum(1 for p in paragraphs if not p.strip())
    if nth > n:
        return False
    paragraph = paragraphs[nth - 1].strip()
    if not paragraph:
        return False
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    got = ""
    for letter in word:
        if letter in {".", ",", "?", "!", "'", '"'}:
            break
        got += letter.lower()
    return n == num_paragraphs and got == first_word.lower()


def _paragraphs(value: str, num: int) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", value)
    n = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip():
            if i in (0, len(paragraphs) - 1):
                n -= 1
            else:
                return False
    return n == num


def _highlights(value: str, num: int) -> bool:
    n = sum(1 for h in re.findall(r"\*[^\n\*]*\*", value) if h.strip("*").strip())
    n += sum(1 for h in re.findall(r"\*\*[^\n\*]*\*\*", value) if h.removeprefix("**").removesuffix("**").strip())
    return n >= num


def _postscript(value: str, marker: str) -> bool:
    value = value.lower()
    if marker == "P.P.S":
        pat = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pat = r"\s*p\.\s?s\..*$"
    else:
        pat = r"\s*" + marker.lower() + r".*$"
    return bool(re.findall(pat, value, flags=re.MULTILINE))


def _json(value: str) -> bool:
    value = (value.strip().removeprefix("```json").removeprefix("```Json").removeprefix("```JSON")
             .removeprefix("```").removesuffix("```").strip())
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def _two_responses(value: str) -> bool:
    valid = []
    parts = value.split("******")
    for i, r in enumerate(parts):
        if not r.strip():
            if i not in (0, len(parts) - 1):
                return False
        else:
            valid.append(r)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _capital_words(value: str) -> int:
    return sum(1 for w in re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", value) if w.isupper())


CHECKS = {
    "keywords:existence": lambda v, k: all(re.search(w, v, flags=re.IGNORECASE) for w in k["keywords"]),
    "keywords:frequency": lambda v, k: _compare(len(re.findall(k["keyword"].strip(), v, flags=re.IGNORECASE)),
                                                k["relation"], k["frequency"]),
    "keywords:forbidden_words": lambda v, k: not any(re.search(r"\b" + w + r"\b", v, flags=re.IGNORECASE)
                                                     for w in k["forbidden_words"]),
    "keywords:letter_frequency": lambda v, k: _compare(collections.Counter(v.lower())[k["letter"]],
                                                       k["let_relation"], k["let_frequency"]),
    "length_constraints:number_sentences": lambda v, k: _compare(count_sentences(v), k["relation"], k["num_sentences"]),
    "length_constraints:number_paragraphs": lambda v, k: _paragraphs(v, k["num_paragraphs"]),
    "length_constraints:number_words": lambda v, k: _compare(count_words(v), k["relation"], k["num_words"]),
    "length_constraints:nth_paragraph_first_word": lambda v, k: _paragraph_first_word(
        v, k["num_paragraphs"], k["nth_paragraph"], k["first_word"]),
    "detectable_content:number_placeholders": lambda v, k: len(re.findall(r"\[.*?\]", v)) >= k["num_placeholders"],
    "detectable_content:postscript": lambda v, k: _postscript(v, k["postscript_marker"].strip()),
    "detectable_format:number_bullet_lists": lambda v, k: (
        len(re.findall(r"^\s*\*[^\*].*$", v, flags=re.MULTILINE))
        + len(re.findall(r"^\s*-.*$", v, flags=re.MULTILINE))) == k["num_bullets"],
    "detectable_format:constrained_response": lambda v, k: any(
        o in v.strip() for o in ("My answer is yes.", "My answer is no.", "My answer is maybe.")),
    "detectable_format:number_highlighted_sections": lambda v, k: _highlights(v, k["num_highlights"]),
    "detectable_format:multiple_sections": lambda v, k: len(
        re.split(r"\s?" + k["section_spliter"] + r"\s?\d+\s?", v)) - 1 >= k["num_sections"],
    "detectable_format:json_format": lambda v, k: _json(v),
    "detectable_format:title": lambda v, k: any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", v)),
    "combination:two_responses": lambda v, k: _two_responses(v),
    "combination:repeat_prompt": lambda v, k: v.strip().lower().startswith(k["prompt_to_repeat"].strip().lower()),
    "startend:end_checker": lambda v, k: v.strip().strip('"').lower().endswith(k["end_phrase"].strip().lower()),
    "startend:quotation": lambda v, k: len(v.strip()) > 1 and v.strip()[0] == '"' and v.strip()[-1] == '"',
    "change_case:capital_word_frequency": lambda v, k: _compare(_capital_words(v), k["capital_relation"],
                                                                k["capital_frequency"]),
    "change_case:english_capital": lambda v, k: v.isupper() and _is_english(v),
    "change_case:english_lowercase": lambda v, k: v.islower() and _is_english(v),
    "punctuation:no_comma": lambda v, k: not re.search(r"\,", v),
}


def check(instruction_id: str, response: str, kwargs: dict) -> bool:
    kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}
    return bool(CHECKS[instruction_id](response, kwargs))


def score(item: dict, response: str) -> dict:
    """Strict prompt-level: every instruction must pass."""
    per = [check(i, response, k) for i, k in zip(item["instruction_id_list"], item["kwargs"])]
    return {"correct": int(all(per)), "instructions": per}
