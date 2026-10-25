# config.py - all model settings in one place

OLLAMA_MODEL = "gemma3:4b"
GEMINI_MODEL = "gemini-flash-lite-latest"   # change to the model your key uses

TEMPERATURE = 0.7        # 0 = precise, 1+ = creative
MAX_TOKENS = 1000        # maximum answer length
STOP_SEQUENCES = ["END"] # stop if the model writes any of these

# Summarization settings
SUMMARY_TEMPERATURE = 0.3   # low = faithful to the text, less invention
SUMMARY_MAX_WORDS = 100     # target summary length
OLLAMA_NUM_CTX = 8192       # how much text Ollama can read at once