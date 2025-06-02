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
    "good evening": "Good evening! How may I help you?",
    "नमस्ते": "नमस्ते! म तपाईंलाई कसरी मद्दत गर्न सक्छु?",
    "कस्तो छ": "म ठिक छु! तपाईंलाई के सहयोग चाहियो?",
    "धन्यवाद": "तपाईंलाई स्वागत छ!",
    "बिदा": "अलविदा! राम्रो दिन बिताउनुहोस्!"
}

def detect_small_talk(query: str) -> str | None:
    normalized = query.lower().strip()
    return SMALL_TALK_RESPONSES.get(normalized)

# Define consistent system behavior for the assistant
SYSTEM_PROMPT = """
You are ClassyBot, a professional virtual assistant for Classic Tech — a leading Internet Service Provider (ISP) and IPTV service provider.
You can understand and respond in both Nepali and English.

IMPORTANT: When responding in Nepali, strictly use Nepali language only. Do NOT respond in Hindi or other languages.
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
    model="gemma3",
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


def is_nepali_text(text: str) -> bool:
    # Check if any character is in the Devanagari Unicode block (Nepali script)
    return any('\u0900' <= ch <= '\u097F' for ch in text)

def chatbot_agent(query: str, session_id: str = "default") -> str:
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

        nepali_query = is_nepali_text(query)

        prompt_prefix = ""
        if nepali_query:
            prompt_prefix = (
            "Please answer the following question strictly in Nepali language only. "
            "Do NOT respond in Hindi or any other language. Use standard Nepali vocabulary and grammar.\n\n"
            )


        question_with_lang = prompt_prefix + query

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": rag_prompt},
        )

        result = qa_chain.invoke({"question": question_with_lang})
        response_text = result.get("answer", "")

        # Remove the prompt prefix from the response if it appears there (just in case)
        if prompt_prefix and response_text.startswith(prompt_prefix):
            response_text = response_text[len(prompt_prefix):].strip()

    except Exception as e:
        return f"Error during RAG processing: {str(e)}"

    if not response_text or any(phrase in response_text.lower() for phrase in fallback_phrases):
        if nepali_query:
            fallback_prompt = f"""
तपाईं ClassyBot हुनुहुन्छ, Classic Tech (नेपालको ISP र IPTV सेवा प्रदायक) को लागि एक ज्ञानमूलक सहायक।

कृपया प्रयोगकर्ताको प्रश्नलाई **सक्दो नेपाली भाषामा मात्र** जवाफ दिनुहोस्।  
हिन्दी वा अन्य भाषामा जवाफ नदिनुहोस्। 

यदि तपाईंलाई उत्तर थाहा छैन भने, कृपया स्पष्ट रूपमा भन्नुहोस् र प्रयोगकर्तालाई https://classic.com.np मा जान वा ग्राहक सहायता सम्पर्क गर्न सुझाव दिनुहोस्।

कृपया अनुमान नलगाउनुहोस्।

प्रयोगकर्ताको प्रश्न: {query}

उत्तर:
"""
        else:
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
