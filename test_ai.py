# Script di prova: legge la chiave API e chiede una risposta a Gemini.

import os
from dotenv import load_dotenv
from google import genai

# Nome del modello Gemini da usare (puoi cambiarlo qui)
MODELLO = "gemini-3.8-flash"

# Carica le variabili dal file .env (nella stessa cartella)
load_dotenv()

# Prende la chiave API dal file .env
chiave_api = os.getenv("GEMINI_API_KEY")

# Crea il client che parla con Gemini
client = genai.Client(api_key=chiave_api)

# Invia la domanda con l'Interactions API (il modo consigliato da Google)
# e riceve la risposta
risposta = client.interactions.create(
    model=MODELLO,
    input="Spiegami in 2 frasi cos'è un'API",
)

# Stampa il testo della risposta sullo schermo
print(risposta.output_text)
