# config.py - all model settings in one place

OLLAMA_MODEL = "gemma3:4b"
GEMINI_MODEL = "gemini-flash-lite-latest"   # change to the model your key uses

TEMPERATURE = 0.7        # 0 = precise, 1+ = creative
MAX_TOKENS = 1000        # maximum answer length
STOP_SEQUENCES = ["END"] # stop if the model writes any of these