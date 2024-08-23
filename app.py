import streamlit as st
from modules import utilities
from modules.config import config
from modules.faiss_db import BookDataBase
from modules.custom_chains import get_novel_continuation_chain


book_info_path = config['book_info_path']
book_info = utilities.load_book_info_json(book_info_path)

embeddings = utilities.get_embeddings()
llm = utilities.get_llm()


def generate_novel_continuation(
        database,
        query,
        length
):
    chain = get_novel_continuation_chain(retriever=database.vector_store.as_retriever(), llm=llm)
    output = chain.invoke(query)
    return output


existing_novels = [book["title"] for book in book_info]

# 侧边栏
with st.sidebar:
    # 滑动条控制续写长度
    st.subheader("续写设置")
    length = st.sidebar.slider("续写长度（字符数）", min_value=100, max_value=1000, value=200, step=50)

    # 选择已有小说
    st.subheader("选择已有小说")
    selected_novel = st.sidebar.selectbox("选择小说", existing_novels)

    # txt文件上传
    st.subheader("上传小说开头文件")
    uploaded_file = st.sidebar.file_uploader("选择一个文本文件", type="txt")

# 设置页面标题
st.title("续梦笔")
st.subheader("轻量级小说续写应用")

if selected_novel is not None:
    book_db = BookDataBase(book_title=selected_novel, book_info_path=book_info_path, embeddings=embeddings)

# 读取上传的文件内容
if uploaded_file is not None:
    if uploaded_file.name in existing_novels:
        bool_db = BookDataBase(book_title=uploaded_file.name, embeddings=embeddings, book_info_path=book_info_path)
    else:
        book_db = BookDataBase(book_title=uploaded_file.name, embeddings=embeddings, book_info_path=book_info_path)
        file_docs = utilities.txt_to_docs(uploaded_file)
        book_db.add_large_documents(file_docs)
else:
    novel_file = ""

# 输入小说开头部分
query = st.text_area("请输入需要的要求", value="", height=200)

# 按钮生成续写
if st.button("一键生成续写内容"):
    if query.strip() == "":
        st.warning("请输入小说开头部分！")
    else:
        novel_continuation = generate_novel_continuation(book_db, query, length)

        # 显示生成的续写内容
        st.subheader("续写结果：")
        st.write(novel_continuation)



