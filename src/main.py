from utils.text import process_pdfs_from_folder
from utils.chatbot import create_vectorstore, create_conversation_chain
from googletrans import Translator

def main():
    folder_path = "src\path"

    chunks = process_pdfs_from_folder(folder_path)
    
    if not chunks:
        print("Nenhum texto extraído dos PDFs.")
        return
    
    vectorstore = create_vectorstore(chunks)

    conversation_chain = create_conversation_chain(vectorstore)

    question = "Oque fala a Seção III - Da Constituição das Comissões?"
    
    response = conversation_chain.run(question)

    translated_response = translate_to_portuguese(response)
    
    print("Resposta:", response)
    print("Resposta em br:", translated_response)

def translate_to_portuguese(text):
    try:
        translator = Translator()
        translated = translator.translate(text, src='auto', dest='pt')
        return translated.text
    except Exception as e:
        print(f"Erro ao traduzir: {e}")
        return text
    
if __name__ == "__main__":
    main()
