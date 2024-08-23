import json
from PIL import Image
from io import BytesIO
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


def txt_to_docs(novel_path):
    loader = TextLoader(novel_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
    )
    docs = text_splitter.split_documents(documents)
    return docs


def display_chain_as_img(intput_chain):
    data = intput_chain.get_graph().draw_png()
    image = Image.open(BytesIO(data))
    image.show()


def load_book_info_json(book_info_path):
    """加载书籍数据"""
    try:
        with open(book_info_path, 'r', encoding='utf-8') as file:
            book_info = json.load(file)
            return book_info
    except Exception as e:
        print(f"书籍数据加载失败：{e}")


def get_embeddings(model="text-embedding-3-small", api_base='https://api.openai.com/v1'):
    return OpenAIEmbeddings(
        model=model,
        openai_api_base=api_base
    )


def get_llm(model='gpt-4o-mini', api_base='https://api.openai.com/v1'):
    return ChatOpenAI(
        model=model,
        openai_api_base=api_base
    )
