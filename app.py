# Pagina web con chat: l'utente carica un documento e fa domande a Gemini.

import os
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

# Nome del modello Gemini da usare per le risposte (stesso di test_ai.py)
MODELLO = "gemini-3.8-flash"

# Modello di embedding gratuito (documentazione ufficiale: ai.google.dev/gemini-api/docs/embeddings)
MODELLO_EMBEDDING = "gemini-embedding-2"

# Quanti frammenti inviare a Gemini per ogni domanda
N_FRAMMENTI = 4

# Quanti testi embeddare insieme, per non superare i limiti del piano gratuito
BLOCCO_EMBEDDING = 16

# Dimensione dei pezzi di testo e quanto si sovrappongono
LUNGHEZZA_FRAMMENTO = 1000
SOVRAPPOSIZIONE = 200

# Gestione del limite gratuito (errore 429): quanto aspettare e quante volte riprovare
PAUSA_SECONDI = 60
MAX_TENTATIVI = 3

# Sopra questo numero di frammenti la preparazione sarà lenta sul piano gratuito
SOGLIA_FRAMMENTI_LENTI = 400

# Istruzione da dare a Gemini quando c'è un documento
ISTRUZIONE_DOCUMENTO = (
    "Rispondi usando solo le informazioni del documento. "
    "Se la risposta non c'è, dillo chiaramente."
)


class LimiteSuperato(Exception):
    """Errore nostro: il limite gratuito è finito anche dopo aver riprovato."""


def e_errore_limite(errore):
    """Dice se l'errore è il 429 (troppe richieste) di Google."""
    testo = str(errore)
    return "429" in testo or "RESOURCE_EXHAUSTED" in testo


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


def dividi_in_frammenti(testo, lunghezza=LUNGHEZZA_FRAMMENTO, sovrapposizione=SOVRAPPOSIZIONE):
    """Divide il testo in pezzi di circa 1000 caratteri, con 200 di sovrapposizione."""
    if not testo:
        return []
    if len(testo) <= lunghezza:
        return [testo]

    frammenti = []
    passo = lunghezza - sovrapposizione
    inizio = 0
    while inizio < len(testo):
        fine = inizio + lunghezza
        frammenti.append(testo[inizio:fine])
        if fine >= len(testo):
            break
        inizio += passo
    return frammenti


def _vettori_da_risposta(risposta):
    """Estrae i numeri (embedding) dalla risposta di Gemini."""
    return [list(e.values) for e in (risposta.embeddings or [])]


def calcola_embeddings(client, testi, per_documento, avanzamento=None, segnaposto=None):
    """Calcola gli embedding a blocchi. Se Google dice 429 (troppe richieste), aspetta e riprova.

    avanzamento: funzione opzionale chiamata dopo ogni blocco con (fatti, totale)
    segnaposto: spazio della pagina (st.empty) dove mostrare il messaggio di attesa
    """
    contenuti = []
    for testo in testi:
        if per_documento:
            # Formato ufficiale per i documenti da cercare (gemini-embedding-2)
            preparato = f"title: none | text: {testo}"
        else:
            # Formato ufficiale per le domande (question answering)
            preparato = f"task: question answering | query: {testo}"
        contenuti.append(
            types.Content(parts=[types.Part.from_text(text=preparato)])
        )

    vettori = []
    totale = len(contenuti)
    for i in range(0, totale, BLOCCO_EMBEDDING):
        blocco = contenuti[i : i + BLOCCO_EMBEDDING]

        # Prova a calcolare questo blocco, fino a MAX_TENTATIVI volte
        for tentativo in range(1, MAX_TENTATIVI + 1):
            try:
                risposta = client.models.embed_content(
                    model=MODELLO_EMBEDDING,
                    contents=blocco,
                )
                break  # Riuscito: esce dai tentativi
            except Exception as errore:
                # Se non è il limite gratuito, è un altro problema: lo segnala subito
                if not e_errore_limite(errore):
                    raise
                # Ultimo tentativo fallito: ci arrendiamo con un messaggio chiaro
                if tentativo == MAX_TENTATIVI:
                    raise LimiteSuperato(
                        "Il limite gratuito di Google è ancora raggiunto dopo "
                        f"{MAX_TENTATIVI} tentativi. Riprova tra qualche minuto."
                    )
                # Altrimenti avvisa l'utente e aspetta prima di riprovare
                if segnaposto is not None:
                    segnaposto.info(
                        f"Limite gratuito raggiunto, attendo {PAUSA_SECONDI} secondi... "
                        f"(tentativo {tentativo} di {MAX_TENTATIVI})"
                    )
                time.sleep(PAUSA_SECONDI)
                if segnaposto is not None:
                    segnaposto.empty()

        vettori.extend(_vettori_da_risposta(risposta))

        # Aggiorna la barra di avanzamento
        if avanzamento is not None:
            avanzamento(min(i + BLOCCO_EMBEDDING, totale), totale)

    return vettori


def frammenti_piu_simili(embedding_domanda, embeddings_frammenti, frammenti, quanti=N_FRAMMENTI):
    """Trova i frammenti più vicini alla domanda con la similarità coseno."""
    if not embeddings_frammenti or not frammenti:
        return []

    matrice = np.array(embeddings_frammenti, dtype=float)
    domanda = np.array(embedding_domanda, dtype=float)
    norme = np.linalg.norm(matrice, axis=1) * (np.linalg.norm(domanda) + 1e-12)
    punteggi = (matrice @ domanda) / (norme + 1e-12)
    quanti = min(quanti, len(frammenti))
    indici = np.argsort(punteggi)[-quanti:][::-1]
    return [frammenti[int(i)] for i in indici]


def mostra_fonti(fonti):
    """Mostra i frammenti usati sotto la risposta."""
    with st.expander("Fonti"):
        for i, pezzo in enumerate(fonti, start=1):
            st.markdown(f"**Frammento {i}**")
            st.write(pezzo)


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
if "frammenti" not in st.session_state:
    st.session_state.frammenti = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = []

# Barra laterale: caricamento del documento
with st.sidebar:
    st.header("Documento")
    file = st.file_uploader(
        "Carica un file PDF o TXT",
        type=["pdf", "txt"],
    )

    # Se l'utente ha scelto un file nuovo, estrai il testo e prepara gli embedding
    # (si fa una sola volta per documento: poi i dati restano in st.session_state)
    if file is not None and file.file_id != st.session_state.id_file:
        testo = estrai_testo(file)
        st.session_state.id_file = file.file_id
        st.session_state.nome_file = file.name
        st.session_state.testo_documento = testo
        st.session_state.frammenti = []
        st.session_state.embeddings = []

        if not testo:
            st.warning(
                "Non è stato possibile estrarre del testo da questo file. "
                "Se è un PDF scansionato (solo immagini), prova un PDF con testo selezionabile oppure un file TXT."
            )
        else:
            frammenti = dividi_in_frammenti(testo)

            # Avviso se il documento è lungo: sul piano gratuito sarà lento
            if len(frammenti) > SOGLIA_FRAMMENTI_LENTI:
                st.warning(
                    f"Documento lungo ({len(frammenti)} frammenti). Sul piano gratuito "
                    "la preparazione sarà lenta: ci possono volere diversi minuti. "
                    "Non chiudere la pagina."
                )

            # Barra di avanzamento e spazio per il messaggio di attesa
            barra = st.progress(0, text="Preparo il documento...")
            avviso_attesa = st.empty()

            def aggiorna_barra(fatti, totale):
                """Aggiorna la barra con quanti frammenti sono stati elaborati."""
                barra.progress(
                    fatti / totale,
                    text=f"Preparo il documento... {fatti} di {totale} frammenti",
                )

            try:
                with st.spinner("Preparo il documento..."):
                    embeddings = calcola_embeddings(
                        client,
                        frammenti,
                        per_documento=True,
                        avanzamento=aggiorna_barra,
                        segnaposto=avviso_attesa,
                    )
                st.session_state.frammenti = frammenti
                st.session_state.embeddings = embeddings
                # A lavoro finito toglie barra e messaggi
                barra.empty()
                avviso_attesa.empty()
            except LimiteSuperato as errore:
                barra.empty()
                avviso_attesa.empty()
                st.error(str(errore))
            except Exception as errore:
                barra.empty()
                avviso_attesa.empty()
                st.error(
                    "Non sono riuscito a preparare il documento per la ricerca. "
                    f"Dettaglio: {errore}"
                )

    # Mostra nome del file e quanti caratteri sono stati letti
    if st.session_state.nome_file:
        n_caratteri = len(st.session_state.testo_documento)
        st.write(f"File: **{st.session_state.nome_file}**")
        st.write(f"Caratteri letti: **{n_caratteri}**")
        if st.session_state.frammenti:
            st.write(f"Frammenti: **{len(st.session_state.frammenti)}**")

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
        if messaggio.get("fonti"):
            mostra_fonti(messaggio["fonti"])

# Casella in basso dove l'utente scrive
domanda = st.chat_input("Scrivi un messaggio")

if domanda:
    # Salva e mostra il messaggio dell'utente
    st.session_state.messaggi.append({"ruolo": "user", "testo": domanda, "fonti": []})
    with st.chat_message("user"):
        st.markdown(domanda)

    fonti = []
    testo_risposta = ""

    # Chiede a Gemini e gestisce eventuali errori
    with st.chat_message("assistant"):
        try:
            # Spazio per un eventuale messaggio di attesa (limite gratuito)
            avviso_domanda = st.empty()
            with st.spinner("Sto pensando..."):
                if st.session_state.frammenti and st.session_state.embeddings:
                    # Cerca i 4 frammenti più simili alla domanda
                    embedding_domanda = calcola_embeddings(
                        client,
                        [domanda],
                        per_documento=False,
                        segnaposto=avviso_domanda,
                    )[0]
                    fonti = frammenti_piu_simili(
                        embedding_domanda,
                        st.session_state.embeddings,
                        st.session_state.frammenti,
                    )
                    pezzi_testo = "\n\n".join(
                        f"Frammento {i}:\n{pezzo}"
                        for i, pezzo in enumerate(fonti, start=1)
                    )
                    contenuto = (
                        f"{ISTRUZIONE_DOCUMENTO}\n\n"
                        f"Documento:\n{pezzi_testo}\n\n"
                        f"Domanda:\n{domanda}"
                    )
                else:
                    # Nessun documento (o preparazione fallita): solo la domanda
                    contenuto = domanda

                risposta = client.interactions.create(
                    model=MODELLO,
                    input=contenuto,
                )
            testo_risposta = risposta.output_text or ""
            st.markdown(testo_risposta)
            if fonti:
                mostra_fonti(fonti)
        except Exception as errore:
            testo_risposta = (
                "Si è verificato un errore durante la richiesta a Gemini. "
                f"Dettaglio: {errore}"
            )
            fonti = []
            st.error(testo_risposta)

        st.session_state.messaggi.append(
            {"ruolo": "assistant", "testo": testo_risposta, "fonti": fonti}
        )