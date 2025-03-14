from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

def create_vectorstore(chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        
        vectorstore.save_local('false_default')

        return vectorstore
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        raise

def create_conversation_chain(vectorstore):
    llm = HuggingFaceEndpoint(
        repo_id="google/gemma-3-1b-it", 
        task='text-generation',
        temperature=0.1,
        # model_kwargs={"max_length": 512}
    )

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    question_prompt = PromptTemplate(
        template="""
        Você é um assistente virtual especializado em direito trabalhista. Sua função é responder perguntas e fornecer explicações detalhadas utilizando exclusivamente as informações extraídas dos documentos PDF carregados, que contêm leis trabalhistas. Siga estas regras rigorosamente:

        1. **Responda SEMPRE em português.**
        2. Seja objetivo e claro em suas respostas. Evite informações desnecessárias e repetições.
        3. **Use APENAS as informações dos documentos carregados para fornecer respostas.**
        4. Se não houver informações suficientes ou se a dúvida não for coberta pelos documentos, responda: "Desculpe, não encontrei informações sobre isso no documento."
        5. **Nunca forneça respostas que não estejam baseadas nos documentos.**

        **Pergunta:** {question}

        **Resposta:**
        """,
        input_variables=["question"]
    )

    question_generator_chain = LLMChain(llm=llm, prompt=question_prompt)

    combine_docs_chain = load_qa_chain(llm, chain_type="stuff")

    conversation_chain = ConversationalRetrievalChain(
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator_chain,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain