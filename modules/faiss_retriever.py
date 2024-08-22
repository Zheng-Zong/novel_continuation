import os
import time
import json
import tiktoken
import faiss
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


class BookRetriever:
    def __init__(self, book_title, embeddings, book_info_path):
        self.book_info_path = book_info_path
        self.book_info = self.__load_book_info_json()
        self.embeddings = embeddings       
        if self.is_book_exists(book_title=book_title):
            self.load_local(book_title=book_title, book_info_path=book_info_path)
            return
        self.book_title = book_title
        self.book_id = max(book["id"] for book in self.book_info) + 1
        
        index = faiss.IndexFlatL2(len(self.embeddings.embed_query("小说数据集")))
        self.vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        self.__update_book_info_json()

    def _find_book(self, book_title=None, book_id=None):
        """查找书籍"""
        for book in self.book_info:
            if (book_title is not None and book_title == book['title']) or \
               (book_id is not None and book_id == book['id']):
                return book
        return None
    
    def __load_book_info_json(self):
        """加载书籍数据"""
        try:
            with open(self.book_info_path, 'r', encoding='utf-8') as file:
                book_info = json.load(file)
                return book_info
        except Exception as e:
            print(f"书籍数据加载失败：{e}")

    def is_book_exists(self, book_title=None, book_id=None):
        """检查书籍是否存在"""
        return self._find_book(book_title, book_id) is not None

    def __update_book_info_json(self):
        """更新书籍信息并保存到 JSON 文件"""
        book_exists = False
        for book in self.book_info:
            if book['id'] == self.book_id:
                book['title'] = self.book_title
                book_exists = True
                break
        
        if not book_exists:
            self.book_info.append({
                "id": self.book_id,
                "title": self.book_title
            })

        with open(self.book_info_path, 'w', encoding='utf-8') as file:
            json.dump(self.book_info, file, ensure_ascii=False, indent=4)

    def load_local(self, book_info_path, book_title=None, book_id=None):
        """加载本地书籍数据"""
        self.book_info_path = book_info_path
        self.book_info = self.__load_book_info_json()
        book = self._find_book(book_title, book_id)
        if book:
            self.book_id = book['id']
            self.book_title = book['title']
            self.vector_store = FAISS.load_local(
                f"./database/{book['id']}",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"成功加载书籍: {self.book_title}")
            return
        else:
            raise ValueError("未找到对应书籍")

    def save_local(self):
        """将当前书籍数据保存到本地"""
        self.vector_store.save_local(f"./database/{self.book_id}")
        self.__update_book_info_json()

    def add_large_documents(self, docs, encoding, TPM_limit=1000000, wait_time=60):
        """添加大文档"""
        tokens_num = 0
        cut_index = 0

        for i, doc in enumerate(docs):
            current_tokens = len(encoding.encode(doc.page_content))
            tokens_num += current_tokens

            if tokens_num > TPM_limit:
                print(f"开始嵌入 {i - cut_index} 个文档，从索引 {cut_index} 到 {i - 1}")
                self.vector_store.add_documents(docs[cut_index:i])
                cut_index = i
                tokens_num = current_tokens
                print(f'部分嵌入完成，等待{wait_time}秒后继续嵌入')
                time.sleep(wait_time)

        if cut_index < len(docs):
            print(f"开始嵌入 {len(docs) - cut_index} 个文档，从索引 {cut_index} 到 {len(docs) - 1}")
            self.vector_store.add_documents(docs[cut_index:])
            print(f'嵌入完成')

    def search(self, query, k=3):
        """根据查询进行相似性搜索，返回前k个结果"""
        results = self.vector_store.similarity_search_with_score(query=query, k=k)
        return results



if __name__ == '__main__':

    book_info_path = 'database/book_info.json'

    try:
        with open(book_info_path, 'r', encoding='utf-8') as file:
            book_info = json.load(file)
    except Exception as e:
        print(f"书籍数据加载失败：{e}")

    TPM_limit = 1000000

    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    loader = TextLoader("novels/TOP/《三体》（实体版1-3全本）作者：刘慈欣.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
    )
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

    book_retriever = BookRetriever("三体", embeddings, book_info, book_info_path)
    # book_retriever.add_large_documents(docs, encoding, TPM_limit=TPM_limit)
    # book_retriever.save_local()
    pass