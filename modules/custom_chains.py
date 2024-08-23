from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from modules.prompts import rag_process_prompt, outline_prompt, expand_template


def get_novel_continuation_chain(retriever, llm):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    first_chain = (
        {"context": retriever | format_docs}
        | rag_process_prompt
        | llm
        | StrOutputParser()
    )
    second_chain = (
        {"context": first_chain, "question": RunnablePassthrough()}
        | outline_prompt
        | llm
        | StrOutputParser()
    )
    output_chain = (
        {"context": second_chain}
        | expand_template
        | llm
        | StrOutputParser()
    )
    return output_chain