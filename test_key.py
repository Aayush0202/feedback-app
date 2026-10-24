from pathlib import Path
import sys
import tomllib
from google import genai

def main():
    # 1. Read GEMINI_API_KEY from .streamlit/secrets.toml using tomllib
    secrets_path = Path(".streamlit") / "secrets.toml"
    if not secrets_path.exists():
        print("Error: .streamlit/secrets.toml not found.")
        sys.exit(1)

    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)

    api_key = secrets.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is missing from secrets.toml.")
        sys.exit(1)

    # 2. Connect to Gemini using the google-genai library
    client = genai.Client(api_key=api_key)

    # 3. Send message using a current Gemini Flash model
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents="Say hello in one short sentence",
        )
        # 4. Print only Gemini's reply (never the API key)
        print("Gemini's reply:")
        print(response.text.strip())
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
