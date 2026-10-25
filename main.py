import os
import time
from dotenv import load_dotenv
import ollama
from google import genai
from google.genai import types, errors
import config

load_dotenv()


def ask_ollama(prompt, temperature=config.TEMPERATURE):
    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": temperature,
            "num_predict": config.MAX_TOKENS,
            "num_ctx": config.OLLAMA_NUM_CTX,
            "stop": config.STOP_SEQUENCES,
        },
    )
    return response["message"]["content"]


def ask_gemini(prompt, temperature=config.TEMPERATURE, retries=3):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=config.MAX_TOKENS,
                    stop_sequences=config.STOP_SEQUENCES,
                ),
            )
            return response.text
        except errors.ServerError:
            print(f"Gemini busy (attempt {attempt}/{retries}), retrying in 10s...")
            time.sleep(10)
    return "Gemini is unavailable right now. Try again later."


if __name__ == "__main__":
    question = "Explain overfitting in 3 sentences."

    print("--- Ollama (local) ---")
    print(ask_ollama(question))

    print("\n--- Gemini (API) ---")
    print(ask_gemini(question))