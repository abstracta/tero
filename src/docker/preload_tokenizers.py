import os
import sys
from transformers import AutoTokenizer

models = os.environ.get("PRELOADED_TOKENIZER_MODELS", "")
if not models:
    sys.exit(0)

for model in models.split(","):
    model = model.strip()
    AutoTokenizer.from_pretrained(model)
    print(f"Preloaded tokenizer for {model}")

