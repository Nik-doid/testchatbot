from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.prompts import PromptTemplate
from .vector_store import get_vector_store
from dotenv import load_dotenv
import os

# Load environment variables (for REDIS_URL)
load_dotenv()

# Define consistent system behavior for the assistant
SYSTEM_PROMPT = """
You are ClassicBot, an intelligent virtual assistant for Classic Tech, a leading Internet Service Provider (ISP) and IPTV provider.
Your job is to assist users with accurate, helpful, and friendly answers to questions related to:

- Internet plans, speeds, and pricing
- IPTV services, packages, and channel availability
- Installation, setup, and troubleshooting for both internet and IPTV
- Billing, renewals, and account-related queries
- Service outages or maintenance updates
- Common customer issues like slow internet, router issues, login problems, or channel errors

Always respond with clarity and professionalism.
If a user asks something outside your knowledge or domain, politely let them know and suggest contacting human support.
Do not make up facts or provide speculative answers.

Use the retrieved documents or knowledge base when possible. If you can't find an answer from documents, use your own general knowledge — but always keep the answer relevant to Classic Tech's services.
"""


# Initialize LLM with the system prompt
llm = OllamaLLM(
    model="mistral",
    base_url="http://localhost:11434",
    temperature=0.7,
    system=SYSTEM_PROMPT
)

# Set up retriever from your vector store (e.g., Chroma)
vector_store = get_vector_store()
retriever = vector_store.as_retriever()

# Custom RAG prompt
rag_prompt_template = """
You are ClassyBot, a helpful assistant for Classic Tech — a leading ISP and IPTV provider.

Use only the provided documents to answer the user’s question.
If you do not find relevant information in the documents, do not try to guess or make up answers.
Instead, politely say: "I'm sorry, I couldn't find relevant information on that."
Avoid mentioning document references explicitly in your reply.

Question: {question}

Relevant Documents:
{context}

Answer:
"""

rag_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=rag_prompt_template
)


def chatbot_agent(query: str, session_id: str = "default") -> str:
    """
    Main function to handle user query with memory, RAG pipeline, and fallback.
    """
    fallback_phrases = ["i don't know", "i'm not sure", "couldn't find", "not listed", "don't have"]

    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        # Set up message history and memory for the session
        message_history = RedisChatMessageHistory(
            url=redis_url,
            session_id=session_id
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True
        )

        # Conversational RAG pipeline with memory + prompt
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": rag_prompt},
        )

        # Run the RAG query
        result = qa_chain({"question": query})
        response_text = result.get("answer", "")

    except Exception as e:
        return f"Error during RAG processing: {str(e)}"

    # Fallback if needed
    if not response_text or any(phrase in response_text.lower() for phrase in fallback_phrases):
        fallback_prompt = f"""
You are ClassyBot, a knowledgeable support assistant for Classic Tech (an ISP and IPTV provider in Nepal).

Please help the user with their question using your best general knowledge.
If you are not sure or the information is not available, say so clearly and recommend the user visit https://classic.com.np or contact customer support.
Do not fabricate information or guess.

User question: {query}

Answer:
"""
        try:
            response_text = llm.invoke(fallback_prompt)
        except Exception as e:
            return f"LLM fallback error: {str(e)}"

    return response_text
