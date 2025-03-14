import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def processor_files(files):
    text = ""
    for file in files:
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text()
    return text

def create_text_chunkc(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1200,
        chunk_overlap=30,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def load_pdfs_from_folder(folder_path):
    files = []
    if not os.path.exists(folder_path):
        print(f"A pasta '{folder_path}' não existe.")
        return []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            files.append(os.path.join(folder_path, filename))
    return files

def process_pdfs_from_folder(folder_path):
    files = load_pdfs_from_folder(folder_path)
    
    if not files:
        print(f"Nenhum arquivo PDF encontrado na pasta '{folder_path}'.")
        return
    
    text = processor_files(files)
    
    chunks = create_text_chunkc(text)
    
    return chunks