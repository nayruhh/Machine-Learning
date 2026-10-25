from pathlib import Path
import config
from main import ask_ollama, ask_gemini


def build_prompt(text, max_words):
    return (
        f"Summarize the following text in at most {max_words} words. "
        "Keep only the key points, use plain language, and do not add "
        "any information that is not in the text.\n\n"
        f"TEXT:\n{text}"
    )


def summarize(text, provider="ollama", max_words=config.SUMMARY_MAX_WORDS):
    prompt = build_prompt(text, max_words)
    if provider == "gemini":
        return ask_gemini(prompt, temperature=config.SUMMARY_TEMPERATURE)
    return ask_ollama(prompt, temperature=config.SUMMARY_TEMPERATURE)


if __name__ == "__main__":
    text = Path("input.txt").read_text(encoding="utf-8")
    print(f"Input: {len(text.split())} words\n")

    print("--- Ollama summary ---")
    print(summarize(text, "ollama"))

    print("\n--- Gemini summary ---")
    print(summarize(text, "gemini"))