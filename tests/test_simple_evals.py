import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from osmium import simple_evals as s  # noqa: E402


def test_answer_extraction_matches_simple_evals():
    assert s.extract_answer("Reasoning...\nAnswer: B") == "B"
    assert s.extract_answer("**Answer: C**") == "C"
    assert s.extract_answer("Answer: $D$") == "D"
    assert s.extract_answer("Answer: \\boxed{A}") == "A"
    assert s.extract_answer("The answer is D") is None
    assert s.extract_answer("উত্তর: ব") == "B"


def test_prompt_and_score():
    row = {"Question": "2+2?", "A": "3", "B": "4", "C": "5", "D": "6", "Answer": "B", "Subject": "college_mathematics"}
    p = s.prompt(row)
    assert p.startswith("Answer the following multiple choice question.") and "B) 4" in p
    r = s.score([row, row], ["Answer: B", "Answer: A"])
    assert r["accuracy"] == 0.5 and r["categories"] == {"stem": 0.5}


@pytest.mark.skipif(not os.environ.get("OSMIUM_DATA"), reason="downloads MMLU (OSMIUM_DATA=1)")
def test_mmlu_file_is_the_pinned_one():
    q = s.questions()
    assert len(q) == 14042 and {r["Answer"] for r in q} == set("ABCD")
    assert {r["Subject"] for r in q} <= set(s.SUBJECT_TO_CATEGORY)
