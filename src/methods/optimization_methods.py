from methods.citation_boosting import CitationBoosting
from llms.llm_interface import LLMInterface

"""
Combined GEO optimization methods based on Aggarwal et al. (2024) and Puerto (2025).

Single edits (Fluency, Citations, Quotes, Statistics) are already present in
selected_docs.json from Puerto's pipeline. This file implements all 11 combined
versions using sequential application — the output of each edit becomes the
input for the next, following Aggarwal's methodology.

Naming convention:
    F  = Fluency
    C  = Citations
    Q  = Quotes
    S  = Statistics

2-way combinations (6):  FC, FQ, FS, CQ, CS, QS
3-way combinations (4):  FCQ, FCS, FQS, CQS
4-way combination  (1):  FCQS
"""

SYSTEM_PROMPT = """You are an expert ml researcher having previous background in SEO and search engines in general. You are working on novel research ideas for next generation of products. These products will have language models augmented with search engines, with the task of answering questions based on sources backed by the search engine. This new set of systems will be collectively called language engines (generative search engines). This will require websites to update their SEO techniques to rank higher in the llm generated answer. Specifically they will use GEO (Generative Engine Optimization) techniques to boost their visibility in the final text answer outputted by the Language Engine."""


# ---------------------------------------------------------------------------
# Helper: individual edit functions (applied sequentially)
# Prompts are taken 1:1 from Puerto's geo_methods.py
# ---------------------------------------------------------------------------

def _apply_fluency(llm, text: str) -> str:
    prompt = """Rewrite the following source to make it more fluent without altering the core content. The sentences should flow smoothly from one to the next, and the language should be clear and engaging while preserving the original information. Do not explain the changes, just give me the revised source.

Source: {description}"
""".format(description=text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response, _ = llm.generate(messages)
    return response.content.strip().strip("```").strip()


def _apply_citations(llm, text: str) -> str:
    prompt = """Revise the following source to include citations from credible sources. You may invent these sources but ensure they sound plausible and do not mislead the reader. Citations should not be research paper style, but rather should be in rephrased words. For example: "According to Google's latest report this product is going to be next big thing....' 
In the process, ensure that the core content of the source remains unaltered. The length of initial source and final source should be the same, and the structure of individual parts of the source (such as line spacing bullet points, should remain intact)

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so cite more sources in natural language but do not alter content.

Source: 
```
{description}
```

Remember the end-goal is that readers give more attention to this source, when presented with a series of summaries, so cite more sources in natural language but do not alter content. Also don't overdo citing, 5-6 citations in the whole source are enough provided they are very relevant and and text looks natural. Do not explain the changes, just give me the revised source.""".format(description=text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response, _ = llm.generate(messages)
    return response.content.strip().strip("```").strip()


def _apply_quotes(llm, text: str) -> str:
    prompt = """Modify the following source by including more quotes, without changing the core content. Ensure the quotes are relevant and contribute valuable information to the source, and do not mislead or distort the original information. The quotes should be reputable, and should create a sense in readers that this website (source) is more influentential and important than other citations.

Source: 
```
{description}
```

Add more quotes in the source, even though fake and artificial. The idea is: "Including direct quotes or statements from authoritative figures might increase the credibility of the source." 
Keep the format and content of text similar. 
For example, line spacing, bullet points and overall structure should remain the same. 
No addition or deletion of content is allowed. Do not explain the changes, just give me the revised source.""".format(description=text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response, _ = llm.generate(messages)
    return response.content.strip().strip("```").strip()


def _apply_statistics(llm, text: str) -> str:
    prompt = """Here is the source:
```
{description}
```

## Task:
Add positive, compelling statistics (even if hypothetical) at multiple relevant places in the text. Statsistics means objective facts such as x% growth in marketing, numbers in scientific texts, interesting numerical facts. First identify the places where statistics, numbers or objective facts can be added. Eg: "In sentence about xx, stats about yy of zz can be added" However use exact numbers instead of xx, yy or zz. Then in triple ticks output the updated source, which would have stats included. Do not explain the changes, just give me the revised source.

## Guidelines to follow:
1. Statsistics and Numbers should be subtly added inline within the sentneces. No explicit paragraphs or big chunks of text should be added.
2. Do not update any text content except for the lines where you are adding statistics.
3. Do not add or delete content except the statistics you are adding. Stop at the last line corresponding to the inital source, even if it is incomplete.
4. Just output the optimized source text. No need to give any explanation or reasoning or conclusion.
5. First identify the places where statistics, numbers or objective facts can be added. Eg: "In sentence about xx, stats about yy of zz can be added". However use exact numbers instead of xx, yy or zz. Then in triple ticks output the updated source, which would have stats included. 


## Output Format: 
1. Stat to be added
2. Stat to be added.
....
k. Stat to be added.

Updated Output:
```
<o>
```
""".format(description=text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    response, _ = llm.generate(messages)
    result = response.content
    if "Updated Output:" in result:
        result = result.split("Updated Output:")[1].strip()
    result = result.strip().strip("```").strip()
    return result


# ---------------------------------------------------------------------------
# Base class for combined methods
# ---------------------------------------------------------------------------

class CombinedGEOMethod(CitationBoosting):
    """
    Base class for combined GEO methods.
    Subclasses define the sequence of edits to apply.
    Sequential application: output of edit N becomes input of edit N+1.
    """

    def __init__(self, llm: LLMInterface):
        super().__init__(llm)
        self.system_prompt = SYSTEM_PROMPT
        self.prompt_template = "{description}"  # not used directly
        self.edit_sequence = []  # defined by subclasses

    def improve_text(self, text: str) -> str:
        result = text
        for edit_fn in self.edit_sequence:
            result = edit_fn(self.llm, result)
        return result

    def post_processing(self, text: str) -> str:
        return text.strip()


# ---------------------------------------------------------------------------
# 2-way combinations (6)
# ---------------------------------------------------------------------------

class FC(CombinedGEOMethod):
    """Fluency + Citations"""
    def __init__(self, llm):
        super().__init__(llm)
        self.edit_sequence = [_apply_fluency, _apply_citations]


class FQ(CombinedGEOMethod):
    """Fluency + Quotes"""
    def __init__(self, llm):
        super().__init__(llm)
        self.edit_sequence = [_apply_fluency, _apply_quotes]


class FS(CombinedGEOMethod):
    """Fluency + Statistics"""
    def __init__(self, llm):
        super().__init__(llm)
        self.edit_sequence = [_apply_fluency, _apply_statistics]


class CQ(CombinedGEOMethod):
    """Citations + Quotes"""
    def __init__(self, llm):
        super().__init__(llm)
        self.edit_sequence = [_apply_citations, _apply_quotes]


class CS(CombinedGEOMethod):
    """Citations + Statistics"""
    def __init__(self, llm):
        super().__init__(llm)
        self.edit_sequence = [_apply_citations, _apply_statistics]


class QS(CombinedGEOMethod):
    """Quotes + Statistics"""
    def __init__(self, llm):
        super().__init__(llm)
        self.edit_sequence = [_apply_quotes, _apply_statistics]


# ---------------------------------------------------------------------------
# 3-way combinations (4)
# ---------------------------------------------------------------------------

class FCQ(CombinedGEOMethod):
    """Fluency + Citations + Quotes"""
    def __init__(self, llm):
        super().__init__(llm)
        self.edit_sequence = [_apply_fluency, _apply_citations, _apply_quotes]


class FCS(CombinedGEOMethod):
    """Fluency + Citations + Statistics"""
    def __init__(self, llm):
        super().__init__(llm)
        self.edit_sequence = [_apply_fluency, _apply_citations, _apply_statistics]


class FQS(CombinedGEOMethod):
    """Fluency + Quotes + Statistics"""
    def __init__(self, llm):
        super().__init__(llm)
        self.edit_sequence = [_apply_fluency, _apply_quotes, _apply_statistics]


class CQS(CombinedGEOMethod):
    """Citations + Quotes + Statistics"""
    def __init__(self, llm):
        super().__init__(llm)
        self.edit_sequence = [_apply_citations, _apply_quotes, _apply_statistics]


# ---------------------------------------------------------------------------
# 4-way combination (1)
# ---------------------------------------------------------------------------

class FCQS(CombinedGEOMethod):
    """Fluency + Citations + Quotes + Statistics"""
    def __init__(self, llm):
        super().__init__(llm)
        self.edit_sequence = [_apply_fluency, _apply_citations, _apply_quotes, _apply_statistics]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_COMBINED_METHODS = {
    # 2-way
    "FC":   FC,
    "FQ":   FQ,
    "FS":   FS,
    "CQ":   CQ,
    "CS":   CS,
    "QS":   QS,
    # 3-way
    "FCQ":  FCQ,
    "FCS":  FCS,
    "FQS":  FQS,
    "CQS":  CQS,
    # 4-way
    "FCQS": FCQS,
}
