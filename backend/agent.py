from langchain_ollama import OllamaLLM
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.prompts import PromptTemplate
from .vector_store import get_vector_store
from dotenv import load_dotenv
import os
from difflib import get_close_matches


# Load environment variables (for REDIS_URL)
load_dotenv()

def detect_small_talk(query: str) -> str | None:
    normalized = query.lower().strip()
    matches = get_close_matches(normalized, SMALL_TALK_RESPONSES.keys(), n=1, cutoff=0.8)
    if matches:
        return SMALL_TALK_RESPONSES[matches[0]]
    return None

SMALL_TALK_RESPONSES = {
    "hello": "Hello! How can I help you today?",
    "hi": "Hi there! What can I assist you with?",
    "hey": "Hey! How can I assist you?",
    "how are you": "I'm doing great! How can I help you today?",
    "thank you": "You're welcome!",
    "thanks": "Glad to help!",
    "bye": "Goodbye! Have a nice day!",
    "good morning": "Good morning! How can I assist you today?",
    "good evening": "Good evening! How may I help you?"
}

def detect_small_talk(query: str) -> str | None:
    normalized = query.lower().strip()
    return SMALL_TALK_RESPONSES.get(normalized)

# Define consistent system behavior for the assistant
SYSTEM_PROMPT = """
You are ClassyBot, a professional virtual assistant for Classic Tech — a leading Internet Service Provider (ISP) and IPTV service provider.

Your job is to assist users with clear, accurate, and friendly responses related to:
- Internet and IPTV plans, pricing, and availability
- Installation, setup, and troubleshooting
- Billing, renewals, and account support
- Service status, outages, and maintenance
- Common issues like slow internet, router problems, login errors, or missing channels

Always respond concisely. Provide detailed help only when asked.
If a question is not related to Classic Tech, respond with:
"Please ask questions related to Classic Tech services."

Never guess or invent information. Use retrieved documents when available; otherwise, respond only if the answer is within your expertise.

"""


# Initialize LLM with the system prompt
llm = OllamaLLM(
    model="mistral",
    base_url="http://localhost:11434",
    temperature=0.7,
    system=SYSTEM_PROMPT
)


vector_store = get_vector_store()
retriever = vector_store.as_retriever()

# Custom RAG prompt
rag_prompt_template = """
You are ClassyBot, a smart and reliable assistant for Classic Tech — a trusted ISP and IPTV provider.

Only use the information in the documents below to answer the user's question.
If no relevant information is found, say:
"I'm sorry, I couldn't find relevant information on that."

Do not mention or refer to the documents directly.

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
    Main function to handle user query with small talk, memory, RAG pipeline, and fallback.
    """

    small_talk_response = detect_small_talk(query)
    if small_talk_response:
        return small_talk_response

    fallback_phrases = ["i don't know", "i'm not sure", "couldn't find", "not listed", "don't have"]

    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        message_history = RedisChatMessageHistory(
            url=redis_url,
            session_id=session_id
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": rag_prompt},
        )

        result = qa_chain({"question": query})
        response_text = result.get("answer", "")

    except Exception as e:
        return f"Error during RAG processing: {str(e)}"

    if not response_text or any(phrase in response_text.lower() for phrase in fallback_phrases):
        fallback_prompt = f"""
You are ClassyBot, a knowledgeable support assistant for Classic Tech (an ISP and IPTV provider in Nepal).

Please help the user with their question using your best general knowledge.

If you're not sure about the answer or the information isn't available, say so clearly and suggest the user visit https://classic.com.np or contact Classic Tech customer support.

Never fabricate or guess information.

User question: {query}

Answer:
"""
        try:
            response_text = llm.invoke(fallback_prompt)
        except Exception as e:
            return f"LLM fallback error: {str(e)}"

    return response_text
