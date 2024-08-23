from modules import templates
from langchain_core.prompts import ChatPromptTemplate

# create the prompt templates
rag_process_prompt = ChatPromptTemplate.from_template(templates.rag_process_template)
outline_prompt = ChatPromptTemplate.from_template(templates.outline_template)
expand_template = ChatPromptTemplate.from_template(templates.expand_template)