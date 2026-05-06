from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load PDF
loader = PyPDFLoader("data/test.pdf")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store in FAISS
db = FAISS.from_documents(chunks, embeddings)

print("✅ PDF processed and stored in vector DB")

# 5. Query loop
while True:
    query = input("\nAsk something (or type 'exit'): ")
    if query.lower() == "exit":
        break

    docs = db.similarity_search(query)

    print("\n📄 Relevant context:")
    for d in docs:
        print("-", d.page_content[:100])