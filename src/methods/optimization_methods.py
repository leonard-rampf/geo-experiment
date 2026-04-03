from methods.citation_boosting import CitationBoosting
from llms.llm_interface import LLMInterface


class SEOOptimization(CitationBoosting):
    """
    Applies SEO-style content modifications to improve retrieval ranking.

    Edits applied:
        1. Semantic Structure and Headings
        2. Keyword Context and Natural Language
        3. Q&A / Query Symmetry

    Goal: Improve the document's relevance signals for keyword-based and semantic retrieval systems.
    """

    def __init__(self, llm: LLMInterface):
        super().__init__(llm)
        self.system_prompt = """You are an expert ml researcher having previous background in SEO and information retrieval. You are working on novel research ideas for next generation search systems, specifically how documents can be optimized to rank higher in modern semantic search engines such as Vertex AI Search. These systems use a combination of lexical matching, semantic understanding, and neural retrieval to rank documents. Your task is to revise source documents by applying targeted on-page SEO edits that improve how these retrieval systems understand and rank the content."""
        self.prompt_template = """Revise the following source to improve its ranking in modern semantic search engines. Apply the following three edits:

1. Semantic Structure and Headings: Write naturally and break up long content logically using descriptive headings where appropriate. Ensure the document has a clear semantic structure that makes it easy for retrieval systems to understand the main topic and subtopics. Do not add HTML tags — use plain text headings and logical paragraph breaks instead.

2. Keyword Context and Natural Language: Provide a complete and comprehensive description of the topic. Anticipate the different words and phrases users might type to find this content and write with those variations in mind — incorporate synonyms, related terms, and natural language variations of the main topic throughout the text. The revised document must stay within +/- 10% of the original word count.

3. Q&A / Query Symmetry: Transform one or two key concepts within the text into a direct Question and Answer format. Explicitly pose a question that a user searching for this topic would likely ask, and immediately provide a dense, direct answer. Embed this naturally into the flow of the text. Do not bloat the document — the output length must stay within +/- 10% of the original word count.

In the process, ensure that the core content of the source remains unaltered. The length and structure of the source should remain similar — line spacing, bullet points, and overall format should stay intact.

Source:
```
{description}
```

Do not explain the changes, just give me the revised source."""

    def post_processing(self, text: str) -> str:
        if "Updated Source:" in text:
            text = text.split("Updated Source:")[-1]
        text = text.strip().strip("```").strip()
        return text


class GEOOptimization(CitationBoosting):
    """
    Applies GEO-style content modifications to improve visibility in LLM-generated responses.

    Edits applied:
        1. Statistics Addition
        2. Quotation and Source Attribution
        3. Authoritative Style

    Goal: Increase the likelihood that the LLM cites this document prominently in its response.
    """

    def __init__(self, llm: LLMInterface):
        super().__init__(llm)
        self.system_prompt = """You are an expert ml researcher having previous background in SEO and search engines in general. You are working on novel research ideas for next generation of products. These products will have language models augmented with search engines, with the task of answering questions based on sources backed by the search engine. This new set of systems will be collectively called language engines (generative search engines). This will require websites to update their techniques to rank higher in the llm generated answer. Specifically they will use GEO (Generative Engine Optimization) techniques to boost their visibility in the final text answer outputted by the Language Engine."""
        self.prompt_template = """Revise the following source to improve its visibility when presented as a search result to a language model. Apply the following three edits:

1. Add relevant statistics and numerical facts at multiple places in the text. These can be hypothetical but must sound realistic and contextually appropriate. Add them inline within existing sentences — no separate paragraphs.

2. Add citations from credible sources in natural language. You may invent these sources but ensure they sound plausible. For example: "According to Google's latest report..." or "A study by Nielsen found that...". Around 4-5 citations in the whole source are enough provided they are relevant and the text looks natural. Do not use research paper style citations.

3. Rewrite the source to sound more confident, expert-like, and authoritative. Replace uncertain or weak phrasing ("might", "could", "possibly", "may be useful") with assertive, definitive language. Present statements as established facts rather than suggestions. The goal is that readers perceive this source as more credible and well-informed than competing sources.

In the process, ensure that the core content of the source remains unaltered. The length and structure of the source should remain the same — line spacing, bullet points, and overall format should stay intact.

Source:
```
{description}
```

Do not explain the changes, just give me the revised source."""

    def post_processing(self, text: str) -> str:
        if "Updated Source:" in text:
            text = text.split("Updated Source:")[-1]
        text = text.strip().strip("```").strip()
        return text


class SEOGEOOptimization(CitationBoosting):
    """
    Applies combined SEO and GEO content modifications.

    Edits applied (GEO first, then SEO):
        1. Statistics Addition
        2. Quotation and Source Attribution
        3. Authoritative Style
        4. Semantic Structure and Headings
        5. Keyword Context and Natural Language
        6. Q&A / Query Symmetry

    Goal: Simultaneously improve retrieval ranking (SEO) and LLM citation visibility (GEO).
    """

    def __init__(self, llm: LLMInterface):
        super().__init__(llm)
        self.system_prompt = """You are an expert ml researcher having previous background in SEO and search engines in general. You are working on novel research ideas for next generation of products. These products will have language models augmented with search engines, with the task of answering questions based on sources backed by the search engine. This new set of systems will be collectively called language engines (generative search engines). Specifically they will use a combination of GEO (Generative Engine Optimization) techniques to boost their visibility in the final text answer outputted by the Language Engine, as well as traditional SEO techniques to rank higher in modern semantic search engines such as Vertex AI Search."""
        self.prompt_template = """Revise the following source to improve both its ranking in modern semantic search engines and its visibility when presented as a search result to a language model. Apply the following six edits:

1. Add relevant statistics and numerical facts at multiple places in the text. These can be hypothetical but must sound realistic and contextually appropriate. Add them inline within existing sentences — no separate paragraphs.

2. Add citations from credible sources in natural language. You may invent these sources but ensure they sound plausible. For example: "According to Google's latest report..." or "A study by Nielsen found that...". Around 4-5 citations in the whole source are enough provided they are relevant and the text looks natural. Do not use research paper style citations.

3. Rewrite the source to sound more confident, expert-like, and authoritative. Replace uncertain or weak phrasing ("might", "could", "possibly", "may be useful") with assertive, definitive language. Present statements as established facts rather than suggestions. The goal is that readers perceive this source as more credible and well-informed than competing sources.

4. Semantic Structure and Headings: Write naturally and break up long content logically using descriptive headings where appropriate. Ensure the document has a clear semantic structure that makes it easy for retrieval systems to understand the main topic and subtopics. Do not add HTML tags — use plain text headings and logical paragraph breaks instead.

5. Keyword Context and Natural Language: Provide a complete and comprehensive description of the topic. Anticipate the different words and phrases users might type to find this content and write with those variations in mind — incorporate synonyms, related terms, and natural language variations of the main topic throughout the text. The revised document must stay within +/- 10% of the original word count.

6. Q&A / Query Symmetry: Transform one or two key concepts within the text into a direct Question and Answer format. Explicitly pose a question that a user searching for this topic would likely ask, and immediately provide a dense, direct answer. Embed this naturally into the flow of the text. Do not bloat the document — the output length must stay within +/- 10% of the original word count.

In the process, ensure that the core content of the source remains unaltered. The length and structure of the source should remain similar — line spacing, bullet points, and overall format should stay intact.

Source:
```
{description}
```

Do not explain the changes, just give me the revised source."""

    def post_processing(self, text: str) -> str:
        if "Updated Source:" in text:
            text = text.split("Updated Source:")[-1]
        text = text.strip().strip("```").strip()
        return text
