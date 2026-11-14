"""Question answering grounded in extracted PDF page text."""

from langchain_core.prompts import PromptTemplate
from utils.llm import get_llm


def generate_document_answer(question, extracted_document):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a question about the document.")
    if not isinstance(extracted_document, dict) or not extracted_document.get("pages"):
        raise ValueError("Extract a text-based PDF before asking questions.")

    page_context = "\n\n".join(
        f"[PAGE {page['page']}]\n{page['text']}"
        for page in extracted_document["pages"]
    )

    prompt = PromptTemplate(
        template="""You answer questions using the supplied PDF text only.

DOCUMENT TEXT:
{document_text}

USER QUESTION:
{question}

RULES:
- Use only information explicitly supported by the document text.
- Do not fill gaps using outside knowledge.
- Cite the source page number for every factual point, using [Page N].
- If the answer is not present, say that the document does not provide enough information.
- If relevant pages have no extracted text, do not assume what they contain.
- If the supplied text is incomplete, disclose that limitation.
- Do not invent quotations, page numbers, figures, or conclusions.
- Be clear and concise.
Return a readable answer with page citations.
""",
        input_variables=["document_text", "question"],
    )

    response = get_llm().invoke(prompt.format(
        document_text=page_context,
        question=question.strip(),
    ))
    content = response.content
    if isinstance(content, str):
        answer = content.strip()
    elif isinstance(content, list):
        answer = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        ).strip()
    else:
        raise ValueError("Unexpected AI response format.")

    if not answer:
        raise ValueError("The AI returned an empty answer.")
    return answer
