from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.prompts import PromptTemplate
from .vector_store import get_vector_store
from dotenv import load_dotenv
import os


llm = OllamaLLM(model="mistral", base_url="http://localhost:11434", temperature=0.7)
vector_store = get_vector_store()
retriever = vector_store.as_retriever()

# Define improved RAG prompt template
rag_prompt_template = """
You are an expert assistant helping a user with their questions.
Use the provided documents to answer accurately, citing facts from them.
If you don't know the answer, say "I couldn't find relevant information."
Answer politely and concisely.

Question: {question}

Relevant Documents:
{context}

Answer:"""

rag_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=rag_prompt_template
)


def chatbot_agent(query: str, session_id: str = "default") -> str:
    fallback_phrases = ["I don't know", "I'm not sure", "can't find"]

    try:
        load_dotenv()
        redis_url = os.getenv("REDIS_URL")

        # Set up message history and memory
        message_history = RedisChatMessageHistory(
            url=redis_url,
            session_id=session_id
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True
        )

        # Create the chain with memory and custom prompt
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": rag_prompt},
        )

        # Query RAG chain
        result = qa_chain({"question": query})
        response_text = result.get("answer", "")

    except Exception as e:
        return f"Error during processing: {str(e)}"

    # If the RAG answer is unsatisfactory, use fallback prompt directly with LLM
    if not response_text or any(phrase.lower() in response_text.lower() for phrase in fallback_phrases):
        fallback_prompt = f"""
You are a knowledgeable and helpful assistant.

Answer the user's question as accurately and completely as possible.
If you are unsure or the information is not available, state that clearly.
Do not guess or fabricate answers.
Be friendly and professional.

User question: {query}
Answer:"""
        try:
            response_text = llm.invoke(fallback_prompt)
        except Exception as e:
            return f"LLM fallback error: {str(e)}"

    return response_text
