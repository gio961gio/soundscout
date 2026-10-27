# Project handoff — SoundScout

## 1. In una frase

SoundScout è una piccola app local-first per sound designer e producer che consente di cercare effetti sonori e sample attraverso descrizioni in linguaggio naturale, invece di dover ricordare nomi di file e strutture di cartelle.

Esempio: l'utente cerca `distant metallic impact with long reverb` e ottiene file audio ordinati per somiglianza semantica, con player, waveform e metadati.

## 2. Perché esiste

Il progetto nasce come portfolio pratico di uno studente del secondo anno di Music Engineering al Politecnico di Milano, con due obiettivi:

1. **Obiettivo candidature:** mostrare capacità di sviluppare un prodotto audio funzionante, capire un problema reale, usare ML audio in modo concreto e valutare i risultati con metodo ingegneristico.
2. **Obiettivo prodotto:** verificare se sound designer, game-audio designer, studi di post-produzione o producer con grandi librerie di sample abbiano un problema abbastanza frequente da giustificare l'uso — e potenzialmente il pagamento — di uno strumento dedicato.

Il successo iniziale non è “aggiungere molte feature”. È dimostrare che, su una libreria reale, SoundScout aiuta una persona a trovare più rapidamente un suono utile.

## 3. Utenti iniziali e problema

### Utente principale ipotizzato

Un sound designer o producer che gestisce centinaia/migliaia di file audio, spesso con nomi poco descrittivi o una tassonomia incoerente.

### Problema ipotizzato

- Si ricorda il suono che cerca, ma non il nome del file.
- Le ricerche testuali tradizionali falliscono quando il nome del file non descrive il contenuto.
- Aprire e ascoltare manualmente molti file è lento e frammenta il flusso creativo.

Queste ipotesi **devono essere verificate con utenti reali**; non sono ancora risultati di ricerca di mercato.

## 4. Proposta di valore

“Trova il suono che hai in mente, non il file che ricordi.”

La prima versione deve essere locale, semplice da provare, rispettosa della privacy dei sample e utile senza richiedere che l'utente rinomini tutta la propria libreria.

## 5. Stato e funzionalità presenti

Il codice attuale include:

- Interfaccia Streamlit (`app.py`).
- Scansione ricorsiva di una cartella locale.
- Supporto dichiarato per WAV, FLAC, OGG, AIFF e tentativo MP3 tramite FFmpeg quando disponibile.
- Estrazione e normalizzazione del segnale audio tramite `soundscout/audio.py`.
- Embedding audio/testo tramite CLAP, checkpoint `laion/clap-htsat-unfused`, in `soundscout/model.py`.
- Cache persistente degli embedding in `~/.soundscout/indexes/`, riutilizzata quando dimensione e mtime del file non sono cambiati.
- Ranking via cosine similarity e filtri per durata, sample rate, formato, canali e cartella di primo livello.
- Anteprima del file audio e waveform.
- Cinque sample sintetici in `demo_audio/` e script `scripts/generate_test_library.py`.
- Test unitari per segmentazione, filtri, ranking e riutilizzo cache.

I punteggi di similarità sono **ranking score, non probabilità calibrate**.

## 6. Stack attuale

- Python
- Streamlit
- NumPy / SciPy
- SoundFile
- PyTorch + Hugging Face Transformers
- CLAP per retrieval testo-audio
- Plotly per waveform
- pytest per test

## 7. Vincoli di prodotto

- Local-first: l'app non deve caricare i sample personali su un server remoto.
- Non introdurre account, cloud storage, database remoti o un backend complesso senza una motivazione verificata.
- Mantieni separati interfaccia, audio I/O, embedding, indicizzazione e ricerca.
- Preferisci feature misurabili e solide a funzioni AI decorative.
- Nessuna promessa di BPM, key detection o timestamp del suono finché non sono implementati e testati.
- Non usare o distribuire sample commerciali di terzi senza diritti. I file demo sono sintetici.

## 8. Criteri di successo iniziali

1. Un nuovo sviluppatore riesce a installare e avviare l'app seguendo il README.
2. L'app indicizza `demo_audio/` e mostra errori comprensibili se un file non è leggibile.
3. Una cartella audio reale può essere indicizzata senza modificare o spostare gli originali.
4. Query testuali restituite come risultati plausibili, da valutare con query e file annotati.
5. La seconda indicizzazione riusa la cache per i file non modificati.
6. Le misure di qualità, tempo di indicizzazione e latenza vengono riportate con hardware e condizioni di test.

## 9. Contesto storico della prima versione

La prima idea del progetto era un MVP in 3–5 giorni: indicizzazione locale, ricerca testuale, preview e filtri. Il codice è già stato generato e i cinque test presenti passano nell'ambiente di preparazione. La prossima fase è consolidarlo, verificare l'esecuzione completa sulla macchina di sviluppo e misurare la qualità prima di ampliare lo scope.
