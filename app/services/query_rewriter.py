from langchain_core.prompts import PromptTemplate

QUERY_REWRITE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
Rewrite the user's question to be clearer and optimized for document search.

Only return the improved query.

Question:
{question}
"""
)


def rewrite_query(llm, question):
    prompt = QUERY_REWRITE_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    return response.content.strip()