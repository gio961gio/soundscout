# Registro iniziale del progetto

## MVP consegnato

- App Streamlit per ricerca testo-audio mediante CLAP.
- Indicizzazione ricorsiva locale.
- Cache incrementale basata su percorso, dimensione e mtime.
- Filtri di metadati, player audio e waveform.
- Cinque file audio sintetici per smoke test.
- Test per segmentazione, filtri, ranking e riuso embedding.
- Documentazione per installazione, architettura e limiti.

## Aggiornamento P2 — Ricerca a Livello di Segmento e Timestamp (27/10/2026)

- Architettura audio temporale: creata dataclass `TimedSegment` e funzione `audio_segments_timed` in `soundscout/audio.py` con hop da 5s su finestre da 10s.
- Modello CLAP: aggiunto `embed_audio_segments` in `soundscout/model.py` per preservare vettori 512-D per ciascun segmento senza diluizione da media globale.
- Indice vettoriale v2: aggiornato `soundscout/index.py` con `INDEX_VERSION = 2`, offset di segmentazione, slicing e salvataggio 2D (`np.vstack`) con supporto robusto al riuso parziale.
- Ranking max-pooling: aggiornato `soundscout/search.py` per identificare il segmento saliente, arricchendo i risultati con `match_start_seconds` e `match_end_seconds`.
- UX interattiva in `app.py`: evidenziazione dell'intervallo temporale rilevato sulla waveform Plotly (overlay arancione trasparente) e auto-seeking dell'audio player su `start_time`.
- Test unitari: aggiunti in `tests/test_segments.py` verifiche per file brevi, file lunghi, max-pooling e concatenazione multi-segmento eterogenea (`18 passed in 2.85s`).
- Revisione tecnica con Argon via `/argon-bridge`: risolte le osservazioni del Round 1 (concatenazione 2D e validazione integrità manifesto `all()`), ottenendo `STATUS: APPROVED`.

## Aggiornamento P1 — Dataset di Valutazione e Benchmark di Retrieval (27/10/2026)

- Creato generatore dataset sintetico `scripts/generate_evaluation_dataset.py`: 36 file audio sintetizzati tramite DSP in 6 categorie professionali (`Impacts`, `Ambiences`, `Foley`, `Tones`, `Mechanical`, `Transitions`).
- Strutturati i file di annotazione ground-truth `evaluation/manifest.json` (36 file con descrizioni e durata) ed `evaluation/queries.json` (20 query classificate per tipologia: material, ambience, foley, tone, mechanical, action).
- Implementato modulo `soundscout/eval.py` per il calcolo delle metriche IR (`recall_at_k`, `precision_at_k`, `reciprocal_rank`, `evaluate_query`, `aggregate_retrieval_metrics`), testato esaustivamente in `tests/test_eval.py` (`13 passed in 2.51s`).
- Creato ed eseguito runner di benchmark `scripts/benchmark_retrieval.py`:
  - **Recall@5: 85.0%** (85% delle query recuperano il target nei primi 5 risultati).
  - **Recall@3: 77.5%**, **Recall@1: 40.0%**.
  - **MRR (Mean Reciprocal Rank): 0.6184**.
  - **Latenza di ricerca:** 78.4 ms mediana, 107.2 ms p95.
  - **Throughput indicizzazione:** 12.23s per 36 file (2.9 file/s su CPU), 125.9 ms con cache incrementale.
- Output benchmark archiviato in `evaluation/benchmark_report.md` e `evaluation/benchmark_summary.json`.
- Supervisione e audit con Argon superata con esito `STATUS: APPROVED` (integrato suggerimento di unione chiavi numeriche resilienti nell'aggregazione).

## Aggiornamento P0 — Completamento e consolidamento iniziale (27/10/2026)

- Risolto blocco ambiente: installato `transformers>=4.44,<5` ed eseguito download iniziale del modello CLAP `laion/clap-htsat-unfused` su CPU.
- Inizializzato repository Git per versionamento e Smart Delta con Argon Bridge.
- Risolto `FutureWarning` in `soundscout/model.py` supportando sia `audio=` che `audios=` per compatibilità con versioni recenti di Transformers.
- Raffinata la gestione degli errori in `soundscout/audio.py` distinguendo fallimenti generici di decodifica da requisiti specifici MP3.
- Espansa la suite di test da 5 a 9 test in `tests/test_audio.py` coprendo directory inesistenti, cartelle vuote, file audio a 0 byte/corrotti e input di ricerca vuoti (`9 passed in 5.18s`).
- Validata indicizzazione e riuso cache su `demo_audio/`: prima passata in 1.86s, re-indicizzazione incrementale in 0.06s (100% riuso).
- Eseguito audit formale tramite `/argon-bridge` con Argon (Gemini Web Pro): revisione approvata (`STATUS: APPROVED`).
