# Assistente Documenti

**Prova l'app online:** https://assistente-documenti-unjuibexr7dbochwebghrd.streamlit.app/

Per usarla serve una chiave Gemini gratuita (si crea su aistudio.google.com) da incollare nella barra laterale. Non viene salvata. Non caricare documenti riservati: sul piano gratuito Google può riutilizzare i contenuti inviati.

## Cosa fa

Carichi un file PDF o TXT e fai domande in chat. L'app risponde usando solo le informazioni del documento e, se la risposta non c'è, lo dice. Sotto ogni risposta si apre la sezione **Fonti**, che mostra i frammenti di testo usati.

## Come funziona (RAG)

1. Il testo viene diviso in frammenti di circa 1000 caratteri, con 200 di sovrapposizione.
2. Ogni frammento viene trasformato in un embedding (una serie di numeri che rappresenta il significato) con il modello `gemini-embedding-2`.
3. Alla domanda, l'app calcola l'embedding della domanda e trova i 4 frammenti più simili con la similarità coseno (numpy).
4. A Gemini vengono inviati solo quei 4 frammenti, con l'istruzione di rispondere solo in base al documento.

## Gestione dei limiti del piano gratuito

- Se Google risponde "troppe richieste" (errore 429), l'app aspetta e riprova da sola, fino a 5 tentativi.
- Se la connessione cade, riprova dopo qualche secondo.
- Una barra di avanzamento mostra quanti frammenti sono stati elaborati.
- In locale i progressi vengono salvati su disco (cartella `cache_embeddings`), così un documento lungo si elabora una sola volta e può ripartire da dove si era fermato.

## Tecnologie usate

- Python
- Streamlit: pagina web con la chat
- Gemini API (google-genai): risposte ed embedding
- numpy: ricerca dei frammenti più simili
- pypdf: lettura del testo dai PDF

## Come ottenere una chiave API gratuita

1. Apri Google AI Studio (aistudio.google.com) e accedi con il tuo account Google.
2. Vai su "Get API key" e crea una chiave.
3. Non condividerla con nessuno e non scriverla nel codice.

Se ti viene chiesta una carta o un piano a pagamento, non procedere. I limiti del piano gratuito possono cambiare: si controllano nella console di AI Studio.

## Installazione su Windows

1. Installa Python da python.org (spunta "Add python.exe to PATH").
2. Scarica il progetto ed entra nella cartella. In PowerShell:

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

3. Apri il file `.env` e scrivi la tua chiave dopo `GEMINI_API_KEY=` (senza spazi né virgolette).

## Avvio

Doppio clic su `avvia_app.bat`, oppure in PowerShell:

```
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Si apre il browser su http://localhost:8501. La finestra nera deve restare aperta finché usi l'app.

## Limiti noti

- Con documenti molto lunghi (libri interi) la preparazione è lenta sul piano gratuito.
- I PDF scansionati (solo immagini) non vengono letti: serve un PDF con testo selezionabile.
- I frammenti sono tagliati a lunghezza fissa, quindi a volte si interrompono a metà parola.

## Idee per migliorarlo

- Taglio dei frammenti a fine frase
- Più documenti insieme, con la fonte di ciascun frammento