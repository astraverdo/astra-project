# 1. Import stuff
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

# 2. Where is my Obsidian vault?
vault_path = Path("/path/to/vault")

# 3. Embedding model will be recalled later
#It takes text and converts it into a long list of numbers that represents the meaning of that text.
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

# 4. Create/load Chroma (vector database)

db_location = "./chroma_langchain_db" # stores my future database in this folder
#creating a chroma object(which is a vector data base) 
vectors = Chroma(
    #parameters
    collection_name="obsidian", #name of collection ak grou[]
    persist_directory=db_location, #where it will store its perminent data
    embedding_function=embeddings #whenever it needs embedding it will use this embedding object
)


# 5. Read Markdown files

documents = [] #brackets are empty python list
ids = []

for file_path in vault_path.rglob("*.md"):
    text = file_path.read_text(encoding="utf-8")

    document = Document(
        page_content=text,
        metadata={"filename": str(file_path.name)},
        id=str(file_path)
    )

    documents.append(document)
    ids.append(str(file_path))


# 6. Put them into Chroma

vectors.add_documents(
    documents=documents,
    ids=ids
)


# 7. Create searcher

retriever = vectors.as_retriever(
    search_kwargs={"k": 5}
)