# Architettura attuale

## Flusso dati

```text
Cartella locale
    │
    ▼
Scansione ricorsiva e lettura metadati
    │
    ├── file invariato ──► riusa embedding dalla cache
    │
    └── file nuovo/modificato
             │
             ▼
       Decodifica audio
             │
             ▼
       Mono / resampling / segmentazione
             │
             ▼
       Embedding audio CLAP
             │
             ▼
       Manifest JSON + embeddings NPY in cache locale

Query testuale ──► embedding testo CLAP ──► cosine similarity
                                                │
                                                ▼
                                         ranking + filtri
                                                │
                                                ▼
                                   Streamlit: player + waveform
```

## File principali

| File | Responsabilità | Note |
|---|---|---|
| `app.py` | UI Streamlit, selezione cartella, pulsante indicizzazione, query, filtri, player e waveform | Mantenerlo leggero; spostare logica di business nei moduli quando serve. |
| `soundscout/audio.py` | Decodifica, metadati, conversione/resampling, segmenti, waveform | Gestisce formati supportati e fallback per alcuni decoder. |
| `soundscout/model.py` | Wrapper CLAP per embedding audio e testo | Usa PyTorch/Transformers e CPU o CUDA quando disponibile. |
| `soundscout/index.py` | Scansione, manifest, cache, aggiornamento incrementale | Cache predefinita in `~/.soundscout/indexes/`. |
| `soundscout/search.py` | Ranking cosine similarity e filtri metadati | La similarità non è una probabilità. |
| `tests/test_audio.py` | Test su segmentazione, filtri, ranking e cache | Ampliare la copertura prima di refactor importanti. |
| `scripts/generate_test_library.py` | Genera 5 file sintetici | Dataset per smoke test, non per valutazione qualità semantica. |

## Dati persistenti

L'indice viene salvato al di fuori della cartella utente, sotto `~/.soundscout/indexes/<hash>/`, in due file principali:

- `manifest.json`: percorsi, metadati, versione dell'indice, modello e record.
- `embeddings.npy`: matrice degli embedding.

Il riuso della cache si basa sul percorso, `size_bytes` e `mtime_ns`, oltre alla versione del modello/indice. Non è un controllo tramite hash completo del contenuto audio.

## Comportamento CLAP

Il checkpoint è `laion/clap-htsat-unfused`. Alla prima esecuzione il pacchetto di modello viene scaricato da Hugging Face se non è già presente nella cache locale. La prima indicizzazione può essere lenta. Le query inglesi sono il punto di partenza consigliato nel README, ma la qualità va misurata nel dominio reale.

Attenzione: l'implementazione attuale media gli embedding dei segmenti audio per creare una rappresentazione per file. È semplice per un MVP, ma può diluire un evento breve all'interno di una registrazione lunga. La roadmap prevede una valutazione e, successivamente, embedding per segmento con restituzione del timestamp rilevante.
