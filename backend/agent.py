import os
import logging
from dotenv import load_dotenv
from langdetect import detect
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.prompts import PromptTemplate
from .vector_store import get_vector_store

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Small Talk Mapping ---
SMALL_TALK_RESPONSES = {
    "hello": "Hello! How can I help you today?",
    "hi": "Hi there! What can I assist you with?",
    "hey": "Hey! How can I assist you?",
    "how are you": "I'm doing great! How can I help you today?",
    "thank you": "You're welcome!",
    "thanks": "Glad to help!",
    "bye": "Goodbye! Have a nice day!",
    "good morning": "Good morning! How can I assist you today?",
    "good evening": "Good evening! How may I help you?",
    "good night": "Good night! Sleep well!",
    "what's up": "Not much! How can I help you?",
    "nice to meet you": "Nice to meet you too! How can I help?",
    "how can you help me": "I’m here to assist you with any questions you have.",
    "are you there": "Yes, I'm here! How can I assist?",
    "i need help": "Sure! Please tell me what you need help with.",
    "thanks for your help": "Anytime! Glad to help.",
}

def detect_small_talk(query: str) -> str | None:
    normalized = query.lower().strip()
    return SMALL_TALK_RESPONSES.get(normalized)

# --- Load API keys & model setup ---
GROQ_API_KEY = os.getenv("GROQ_API")
REDIS_URL = os.getenv("REDIS_URL")

llm = ChatGroq(
    model_name="llama3-70b-8192",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

vector_store = get_vector_store()
retriever = vector_store.as_retriever()

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

# --- Main Agent Function ---
def chatbot_agent(query: str, session_id: str = "default") -> str:
    small_talk_response = detect_small_talk(query)
    if small_talk_response:
        return small_talk_response

    try:
        language = detect(query)
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        language = "en"

    fallback_phrases = [
        "i don't know", "i'm not sure", "couldn't find", "not listed", "don't have",
        "मलाई थाहा छैन", "म निश्चित छैन", "पत्ता लगाउन सकिएन"
    ]

    try:
        message_history = RedisChatMessageHistory(
            url=REDIS_URL,
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

        result = qa_chain.invoke({"question": query})
        response_text = result.get("answer", "")

        if hasattr(response_text, 'content'):
            response_text = response_text.content

    except Exception as e:
        logger.error(f"RAG processing error: {e}")
        return "Sorry, I encountered an issue processing your request. Please try again later."

    # --- Fallback to general LLM answer if RAG is not helpful ---
    if not response_text or any(phrase in response_text.lower() for phrase in fallback_phrases):
        fallback_prompt = (
            f"""
तपाईं ClassyBot हुनुहुन्छ, Classic Tech (नेपालको इन्टरनेट र आइपिटिभी सेवा प्रदायक) को लागि जानकार सहायक।

कृपया प्रयोगकर्ताको सोधिएको प्रश्नलाई तपाईंको ज्ञानको भरमा नेपाली भाषामा उत्तर दिनुहोस्।

यदि तपाईंलाई उत्तर थाहा छैन भने वा सोधिएको जानकारी उपलब्ध छैन भने स्पष्ट रूपमा भन्नुहोस् र प्रयोगकर्तालाई https://classic.com.np मा जान वा ग्राहक सेवा सम्पर्क गर्न सुझाव दिनुहोस्।

प्रश्न: {query}

उत्तर:
"""
            if language == "ne" else
            f"""
You are ClassyBot, a knowledgeable support assistant for Classic Tech (an ISP and IPTV provider in Nepal).

Please help the user with their question using your best general knowledge.

If you're not sure about the answer or the information isn't available, say so clearly and suggest the user visit https://classic.com.np or contact Classic Tech customer support.

Never fabricate or guess information.

User question: {query}

Answer:
"""
        )

        try:
            response_text = llm.invoke(fallback_prompt)
            if hasattr(response_text, 'content'):
                response_text = response_text.content
        except Exception as e:
            logger.error(f"LLM fallback error: {e}")
            return "Sorry, I'm unable to answer that at the moment. Please try again later."

    return response_text
