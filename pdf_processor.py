import os, tempfile, shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

def process_uploaded_pdf(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    shutil.rmtree(temp_dir)
    return documents

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
