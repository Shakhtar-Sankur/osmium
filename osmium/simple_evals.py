"""MMLU exactly as OpenAI's simple-evals runs it, which Apple's 2025 report used for its
on-device model (arXiv 2507.13575, Tables 1 and 3): zero-shot, the model reasons and ends with
"Answer: X". The prompt, the answer patterns, the response normalisation and the subject
categories below are copied from openai/simple-evals at commit 652c89d (common.py, mmlu_eval.py).

simple-evals: Copyright (c) 2024 OpenAI, MIT License.
"""
import csv
import hashlib
import os
import re
import urllib.request

MMLU_URL = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
MMLU_SHA256 = "15b6785d49e0012602e089558a7a0dfb916baf97e9295aa25b48062f13c6afbb"
CACHE = os.environ.get("OSMIUM_CACHE", os.path.expanduser("~/.cache/osmium"))

QUERY_TEMPLATE_MULTICHOICE = "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"

MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = '(?i){}[ \t]*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])'

MULTILINGUAL_ANSWER_REGEXES = [
    'Answer\\s*:',
    'Answer\\s*:\u200b\u200b\u200b\u200b\u200b\u200b',
    'উত্তর\\s*:',
    'उत्तर\\s*:',
    'উত্তরঃ',
    'উত্তর\\s*:',
    'Antwort\\s*:',
    '답변\\s*:',
    '정답\\s*:',
    '답\\s*:',
    '答案\\s*：',
    '答案\\s*:',
    '答\\s*：',
    '答\\s*:',
    '答复\\s*：',
    '答曰\\s*：',
    'الإجابة:',
    'الجواب:',
    'إجابة:',
    'الإجابة النهائية:',
    'الإجابة الصحيحة:',
    'الإجابة الصحيحة هي:',
    'الإجابة هي:',
    'الجواب النهائي:',
    'Respuesta\\s*:',
    'Risposta\\s*:',
    '答え\\s*:',
    '答え\\s*：',
    '回答\\s*:',
    '回答\\s*：',
    '解答\\s*:',
    'Jawaban\\s*:',
    'Réponse\\s*:',
    'Resposta\\s*:',
    'Jibu\\s*:',
    'Idahun\\s*:',
    'Ìdáhùn\\s*:',
    'Idáhùn\\s*:',
    'Àmọ̀nà\\s*:',
    'Àdáhùn\\s*:',
    'Ànúgọ\\s*:',
    'Àṣàyàn\\s*:',
]

SUBJECT_TO_CATEGORY = {
    'abstract_algebra': 'stem',
    'anatomy': 'other',
    'astronomy': 'stem',
    'business_ethics': 'other',
    'clinical_knowledge': 'other',
    'college_biology': 'stem',
    'college_chemistry': 'stem',
    'college_computer_science': 'stem',
    'college_mathematics': 'stem',
    'college_medicine': 'other',
    'college_physics': 'stem',
    'computer_security': 'stem',
    'conceptual_physics': 'stem',
    'econometrics': 'social_sciences',
    'electrical_engineering': 'stem',
    'elementary_mathematics': 'stem',
    'formal_logic': 'humanities',
    'global_facts': 'other',
    'high_school_biology': 'stem',
    'high_school_chemistry': 'stem',
    'high_school_computer_science': 'stem',
    'high_school_european_history': 'humanities',
    'high_school_geography': 'social_sciences',
    'high_school_government_and_politics': 'social_sciences',
    'high_school_macroeconomics': 'social_sciences',
    'high_school_mathematics': 'stem',
    'high_school_microeconomics': 'social_sciences',
    'high_school_physics': 'stem',
    'high_school_psychology': 'social_sciences',
    'high_school_statistics': 'stem',
    'high_school_us_history': 'humanities',
    'high_school_world_history': 'humanities',
    'human_aging': 'other',
    'human_sexuality': 'social_sciences',
    'international_law': 'humanities',
    'jurisprudence': 'humanities',
    'logical_fallacies': 'humanities',
    'machine_learning': 'stem',
    'management': 'other',
    'marketing': 'other',
    'medical_genetics': 'other',
    'miscellaneous': 'other',
    'moral_disputes': 'humanities',
    'moral_scenarios': 'humanities',
    'nutrition': 'other',
    'philosophy': 'humanities',
    'prehistory': 'humanities',
    'professional_accounting': 'other',
    'professional_law': 'humanities',
    'professional_medicine': 'other',
    'professional_psychology': 'social_sciences',
    'public_relations': 'social_sciences',
    'security_studies': 'social_sciences',
    'sociology': 'social_sciences',
    'us_foreign_policy': 'social_sciences',
    'virology': 'other',
    'world_religions': 'humanities',
}


def normalize_response(response):
    """Remove markdown and LaTeX that may prevent a match (simple-evals common.normalize_response)."""
    return (
        response.replace("**", "")
        .replace("$\\boxed{", "")
        .replace("}$", "")
        .replace("\\$", "")
        .replace("$\\text{", "")
        .replace("$", "")
        .replace("\\mathrm{", "")
        .replace("\\{", "")
        .replace("\\text", "")
        .replace("\\(", "")
        .replace("\\mathbf{", "")
        .replace("{", "")
        .replace("\\boxed", "")
    )


def normalize_extracted_answer(answer):
    """simple-evals common.normalize_extracted_answer."""
    return (
        answer.replace("أ", " A").replace("ب", " B").replace("ج", " C").replace("د", " D")
        .replace("অ", " A").replace("ব", " B").replace("ড", " C").replace("ঢ", " D")
        .replace("Ａ", " A").replace("Ｂ", " B").replace("Ｃ", " C").replace("Ｄ", " D")
        .strip()
    )


def extract_answer(response):
    """The letter simple-evals reads from a response, or None."""
    text = normalize_response(response)
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        match = re.search(MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex), text)
        if match:
            return normalize_extracted_answer(match.group(1))
    return None


def prompt(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


def questions():
    """The 14,042 MMLU test questions in simple-evals' file, checked against a pinned hash."""
    path = os.path.join(CACHE, "mmlu.csv")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        urllib.request.urlretrieve(MMLU_URL, path + ".part")
        os.replace(path + ".part", path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if digest != MMLU_SHA256:
        raise RuntimeError(f"{path}: sha256 {digest}, expected {MMLU_SHA256}")
    with open(path, encoding="utf-8", newline="") as f:
        return [{k: v for k, v in r.items() if k} for r in csv.DictReader(f)]


def score(rows, responses):
    """Accuracy overall and by simple-evals category, as simple-evals aggregates it."""
    by = {}
    correct = 0
    for row, response in zip(rows, responses):
        ok = extract_answer(response) == row["Answer"]
        correct += ok
        cat = SUBJECT_TO_CATEGORY.get(row["Subject"], "other")
        n, c = by.get(cat, (0, 0))
        by[cat] = (n + 1, c + ok)
    return {"accuracy": correct / len(rows), "questions": len(rows),
            "categories": {k: round(c / n, 4) for k, (n, c) in sorted(by.items())}}
