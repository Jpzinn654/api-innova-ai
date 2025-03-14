from utils.text import process_pdfs_from_folder
from utils.chatbot import create_vectorstore, create_conversation_chain

def main():
    folder_path = "src\path"

    chunks = process_pdfs_from_folder(folder_path)
    
    if not chunks:
        print("Nenhum texto extraído dos PDFs.")
        return
    
    vectorstore = create_vectorstore(chunks)

    conversation_chain = create_conversation_chain(vectorstore)

    question = "Qual é a legislação sobre férias no Brasil?"
    
    response = conversation_chain.run(question)
    
    print("Resposta:", response)
if __name__ == "__main__":
    main()
