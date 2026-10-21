# Pagina web con chat: l'utente carica un documento e fa domande a Gemini.

import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

# Nome del modello Gemini da usare (stesso di test_ai.py)
MODELLO = "gemini-3.8-flash"

# Se il documento è più lungo di così, usiamo solo l'inizio
MAX_CARATTERI = 100000

# Istruzione da dare a Gemini quando c'è un documento
ISTRUZIONE_DOCUMENTO = (
    "Rispondi usando solo le informazioni del documento. "
    "Se la risposta non c'è nel documento, dillo chiaramente."
)


def estrai_testo(file):
    """Legge il testo dal file caricato. Per i PDF usa pypdf, per i TXT una lettura semplice."""
    nome = file.name.lower()

    if nome.endswith(".pdf"):
        # Apre il PDF e unisce il testo di tutte le pagine
        lettore = PdfReader(file)
        pezzi = []
        for pagina in lettore.pages:
            pezzi.append(pagina.extract_text() or "")
        return "\n".join(pezzi).strip()

    if nome.endswith(".txt"):
        # Legge il file di testo come UTF-8
        return file.read().decode("utf-8", errors="replace").strip()

    return ""


# Carica le variabili dal file .env (nella stessa cartella)
load_dotenv()

# Prende la chiave API dal file .env
chiave_api = os.getenv("GEMINI_API_KEY")

# Crea il client che parla con Gemini
client = genai.Client(api_key=chiave_api)

# Titolo e breve spiegazione in cima alla pagina
st.title("Assistente Documenti")
st.write("Carica un PDF o un file di testo e fai domande sul suo contenuto")

# Se è la prima volta che apri la pagina, prepara i contenitori vuoti
if "messaggi" not in st.session_state:
    st.session_state.messaggi = []
if "testo_documento" not in st.session_state:
    st.session_state.testo_documento = ""
if "nome_file" not in st.session_state:
    st.session_state.nome_file = None
if "id_file" not in st.session_state:
    st.session_state.id_file = None

# Barra laterale: caricamento del documento
with st.sidebar:
    st.header("Documento")
    file = st.file_uploader(
        "Carica un file PDF o TXT",
        type=["pdf", "txt"],
    )

    # Se l'utente ha scelto un file nuovo, estrai il testo
    if file is not None and file.file_id != st.session_state.id_file:
        testo = estrai_testo(file)
        st.session_state.id_file = file.file_id
        st.session_state.nome_file = file.name
        st.session_state.testo_documento = testo

        if not testo:
            st.warning(
                "Non è stato possibile estrarre del testo da questo file. "
                "Se è un PDF scansionato (solo immagini), prova un PDF con testo selezionabile oppure un file TXT."
            )

    # Mostra nome del file e quanti caratteri sono stati letti
    if st.session_state.nome_file:
        n_caratteri = len(st.session_state.testo_documento)
        st.write(f"File: **{st.session_state.nome_file}**")
        st.write(f"Caratteri letti: **{n_caratteri}**")

        if n_caratteri > MAX_CARATTERI:
            st.warning(
                f"Il documento è lungo ({n_caratteri} caratteri). "
                f"Userò solo i primi {MAX_CARATTERI} caratteri."
            )

    # Svuota solo la chat: il documento resta caricato
    if st.button("Nuova conversazione"):
        st.session_state.messaggi = []
        st.rerun()

# Se non c'è un documento, invita a caricarlo nella barra laterale
if not st.session_state.nome_file:
    st.info("Carica un PDF o un file TXT nella barra laterale per iniziare.")

# Mostra tutti i messaggi già scambiati (utente e assistente)
for messaggio in st.session_state.messaggi:
    with st.chat_message(messaggio["ruolo"]):
        st.markdown(messaggio["testo"])

# Casella in basso dove l'utente scrive
domanda = st.chat_input("Scrivi un messaggio")

if domanda:
    # Salva e mostra il messaggio dell'utente
    st.session_state.messaggi.append({"ruolo": "user", "testo": domanda})
    with st.chat_message("user"):
        st.markdown(domanda)

    # Prepara il testo da inviare a Gemini
    testo_documento = st.session_state.testo_documento
    if testo_documento and len(testo_documento) > MAX_CARATTERI:
        testo_documento = testo_documento[:MAX_CARATTERI]

    if testo_documento:
        # C'è un documento: domanda + testo + istruzione
        contenuto = (
            f"{ISTRUZIONE_DOCUMENTO}\n\n"
            f"Documento:\n{testo_documento}\n\n"
            f"Domanda:\n{domanda}"
        )
    else:
        # Nessun documento: solo la domanda, come prima
        contenuto = domanda

    # Chiede a Gemini e gestisce eventuali errori
    with st.chat_message("assistant"):
        try:
            with st.spinner("Sto pensando..."):
                risposta = client.interactions.create(
                    model=MODELLO,
                    input=contenuto,
                )
            testo_risposta = risposta.output_text or ""
        except Exception as errore:
            testo_risposta = (
                "Si è verificato un errore durante la richiesta a Gemini. "
                f"Dettaglio: {errore}"
            )
            st.error(testo_risposta)

        if not testo_risposta.startswith("Si è verificato un errore"):
            st.markdown(testo_risposta)

        st.session_state.messaggi.append({"ruolo": "assistant", "testo": testo_risposta})
