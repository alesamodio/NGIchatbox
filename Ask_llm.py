import openai
import os

from config import VECTORSTORE_PATH, SIMILARITY_THRESHOLD, MAX_CHUNKS, EXTRA_CHUNKS

# Set OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

def query_llm(context, user_query):
    """Query OpenAI's GPT model using the retrieved context."""
    prompt = f"""
    You are a highly knowledgeable assistant.
    The user has asked:
    "{user_query}"
    
    Below is the retrieved context from trusted documents:
    {context}
    
    Provide a clear and complete response based primarily on the given context. If more information is needed, indicate what additional details should be retrieved.
    
    If you need more information, explicitly say "NEED MORE INFO: YES" at the end.
    Also, suggest a more specific search query that could help find the missing details.

    Otherwise, say "NEED MORE INFO: NO".
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response['choices'][0]['message']['content']

def ask_user_for_query_refinement(suggested_query):
    """Returns the suggested query for UI to handle refinement logic."""
    return {
        "needs_refinement": True,
        "suggested_query": suggested_query
    }
    