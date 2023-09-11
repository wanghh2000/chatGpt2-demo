# LangChain实现基于本地私有知识库问答的流程：
# https://blog.csdn.net/qq_36187610/article/details/131900517

# 加载文件 → 读取文件 → 文本分割 → 文本向量化 → 问题向量化 → 在文本向量中匹配与问题最相近的TopK → 匹配出的文本作为上下文与问题一起添加进prompt → 提交给LLM生成回答

# pip install langchain docx2txt sentence_transformers

# 1. 载入本地文档，并切片成若干小片段
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = Docx2txtLoader("C:/tmp/aaa.docx")
data = loader.load()

# 初始化加载器 chunk_size是片段长度，chunk_overlap是片段之间的重叠长度
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=128)

# 切割加载的 document
split_docs = text_splitter.split_documents(data)

print(len(split_docs))
print(split_docs[0])

# 2. 基于开源的预训练的Embedding语言模型，对文本进行向量化

from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import sentence_transformers

# EMBEDDING_MODEL = "C:/tmp/text2vec_ernie/"
EMBEDDING_MODEL = "nghuyong/ernie-3.0-base-zh"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
embeddings.client = sentence_transformers.SentenceTransformer(embeddings.model_name, device='cuda')

# 3. 将切分好的文本片段转换为向量，并存入FAISS中
from langchain.vectorstores import FAISS

db = FAISS.from_documents(split_docs, embeddings)
# db.save_local("/workdir/temp/faiss/") # 指定Faiss的位置
db.save_local("C:/tmp/faiss/") # 指定Faiss的位置

# 4. 载入FAISS数据库
# db = FAISS.load_local("/workdir/temp/chroma/",embeddings=embeddings)
db = FAISS.load_local("C:/tmp/faiss/",embeddings=embeddings)

# 5. 将问题也转换为文本向量，并在FAISS中查找最为相近的TopK
question = "新能源行业发展了多久？"
similarDocs = db.similarity_search(question, include_metadata=True, k=2)
for x in similarDocs:
    print(x)

# 6. 直接调用LangChain的RetrievalQA，实现基于上下文的问答。省去了写prompt的步骤
# from langchain.chains import RetrievalQA
# import IPython

# retriever = db.as_retriever()
# qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
# query = "新能源行业发展了多久？"
# print(qa.run(query))
