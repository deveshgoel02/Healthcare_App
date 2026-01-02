import os
from dotenv import load_dotenv
from groq import Groq

# -------------------------------
# ENV SETUP
# -------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. "
        "Please add it to your .env file or environment variables."
    )

# -------------------------------
# GROQ CLIENT
# -------------------------------

client = Groq(api_key=GROQ_API_KEY)

# -------------------------------
# LIST MODELS
# -------------------------------

try:
    models = client.models.list()

    print("\nAvailable Groq Models:\n" + "-" * 30)

    for model in getattr(models, "data", []):
        model_id = getattr(model, "id", None)
        if model_id:
            print(f"- {model_id}")

    print("\nTotal models:", len(models.data))

except Exception as e:
    print("❌ Error listing models:")
    print(str(e))
