# Assistente Documenti

**Prova l'app online:** https://assistente-documenti-unjuibexr7dbochwebghrd.streamlit.app/

Per usarla serve una chiave Gemini gratuita (si crea su aistudio.google.com) da incollare nella barra laterale. Non caricare documenti riservati.

Un programma semplice per fare domande su un documento.

Carichi un file PDF o TXT e scrivi in chat. Il programma legge il testo del documento e chiede una risposta a Gemini, usando solo le informazioni contenute nel file.

## Tecnologie usate

- **Python** — linguaggio del programma
- **Streamlit** — pagina web con la chat
- **Gemini API** — intelligenza artificiale di Google che risponde alle domande
- **pypdf** — legge il testo dai file PDF

## Come ottenere una chiave API gratuita

1. Apri [Google AI Studio](https://aistudio.google.com).
2. Accedi con un account Google.
3. Crea una chiave API (API key).
4. Copiala e tienila per te: è come una password.

## Come installare e avviare su Windows

Apri **PowerShell** e vai nella cartella del progetto. Esempio:

```powershell
cd C:\assistente-documenti
```

### 1. Crea l'ambiente virtuale

```powershell
python -m venv .venv
```

### 2. Installa i pacchetti

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Crea il file della chiave

Copia `.env.example` e rinominalo in `.env`. Poi apri `.env` e metti la tua chiave al posto di `inserisci_qui_la_tua_chiave`.

In PowerShell:

```powershell
copy .env.example .env
```

### 4. Avvia l'app

Puoi fare doppio clic su `avvia_app.bat` nella cartella del progetto: apre l'app senza usare PowerShell.

In alternativa, da PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Si apre nel browser, di solito all'indirizzo `http://localhost:8501`.

Nella barra laterale carica un PDF o un TXT, poi scrivi una domanda nella chat. Il pulsante **Nuova conversazione** cancella i messaggi ma lascia il documento caricato.
