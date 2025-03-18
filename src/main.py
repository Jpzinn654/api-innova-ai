from fastapi import FastAPI
from pydantic import BaseModel
from utils.text import process_pdfs_from_folder
from utils.chatbot import create_vectorstore, create_conversation_chain
from googletrans import Translator

app = FastAPI()

# Variáveis globais
folder_path = "src\path"
chunks = process_pdfs_from_folder(folder_path)

# Checa se os PDFs foram processados corretamente
if not chunks:
    raise ValueError("Nenhum texto extraído dos PDFs.")

# Cria o vetor de armazenamento e a cadeia de conversa
vectorstore = create_vectorstore(chunks)
conversation_chain = create_conversation_chain(vectorstore)

class QuestionRequest(BaseModel):
    question: str

def translate_to_portuguese(text):
    try:
        translator = Translator()
        translated = translator.translate(text, src='auto', dest='pt')
        return translated.text
    except Exception as e:
        print(f"Erro ao traduzir: {e}")
        return text

@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    question = request.question
    
    # Gera a resposta com base na pergunta
    response = conversation_chain.run(question)
    
    # Traduz a resposta para o português
    translated_response = translate_to_portuguese(response)
    
    return {
        "response_pt": translated_response
    }

@app.get("/")
async def root():
    return {"message": "API de Perguntas para Chatbot"}

