# Roadmap — priorità e definizione di done

La roadmap è intenzionalmente sequenziale: prima stabilità e misure, poi feature. Non implementare tutto in un unico passaggio.

## P0 — Avvio end-to-end e robustezza (completata)

**Obiettivo:** confermare che il repository parta pulito sulla macchina di sviluppo.

- [x] Configurare ambiente Python (`3.11.5`), installare `transformers>=4.44,<5` e verificare tutte le dipendenze.
- [x] Inizializzare repository Git per tracciamento versioni e Smart Delta audit (`argon-bridge`).
- [x] Eseguire `python -m pytest -q`: estesa la suite a 9 test unitari (tutti passati).
- [x] Caricare e testare il checkpoint CLAP (`laion/clap-htsat-unfused`) con fallback di compatibilità per processore audio.
- [x] Indicizzare `demo_audio/` e misurare prima indicizzazione (1.86s per 5 file) e re-index da cache (0.06s, riuso 100%).
- [x] Verificare query reali con score di similarità su suoni metallici, toni puri e rumble.
- [x] Testare e gestire percorsi inesistenti, cartelle vuote e file audio corrotti (isolamento errori senza blocco del processo).
- [x] Audit e supervisione tecnica superata con **Argon** via `/argon-bridge` (`STATUS: APPROVED`).

**Done:** installazione, smoke test ed edge-case verificati e documentati.

## P1 — Dataset di valutazione e baseline di retrieval (completata)

**Obiettivo:** capire se il tool trova davvero i suoni giusti con metriche scientifiche e dataset riproducibile.

- [x] Generato dataset sintetico parametrico di 36 suoni WAV (48 kHz) in 6 categorie (`Impacts`, `Ambiences`, `Foley`, `Tones`, `Mechanical`, `Transitions`) tramite `scripts/generate_evaluation_dataset.py`.
- [x] Definito ground-truth formalizzato in `evaluation/manifest.json` e 20 query descrittive etichettate per categoria in `evaluation/queries.json`.
- [x] Creato modulo `soundscout/eval.py` con calcolo rigoroso di Recall@K, Precision@K e MRR (Mean Reciprocal Rank), con test dedicati in `tests/test_eval.py`.
- [x] Eseguito benchmark automatico (`scripts/benchmark_retrieval.py`):
  - **Recall@1:** 40.0%
  - **Recall@3:** 77.5%
  - **Recall@5:** 85.0%
  - **Precision@1:** 40.0%
  - **Precision@5:** 18.0%
  - **MRR:** 0.6184
  - **Latenza:** 78.4 ms (p95: 107.2 ms)
  - **Throughput indicizzazione:** 2.9 file/s su CPU (prima passata: 12.23s, seconda passata in cache: 125.9 ms).
- [x] Generati report dettagliati in `evaluation/benchmark_report.md` e `evaluation/benchmark_summary.json` con analisi qualitativa di successi ed errori.
- [x] Supervisione e approvazione tecnica di **Argon** via `/argon-bridge` (`STATUS: APPROVED`).

**Done:** metriche reali riproducibili, latenze misurate e casi d'errore documentati.

## P2 — Ricerca a livello di segmento e timestamp

**Obiettivo:** migliorare la ricerca in registrazioni lunghe.

- [ ] Salvare un embedding per segmento (con timestamp iniziale/finale) invece di una sola media per file.
- [ ] Restituire il miglior segmento e il relativo intervallo temporale nel file.
- [ ] Consentire l'ascolto del file dal punto del match, se fattibile nell'UI attuale.
- [ ] Conservare un ranking a livello file per evitare risultati duplicati e rendere chiaro il criterio di aggregazione.
- [ ] Aggiungere test per timestamp, file brevi e file lunghi.

**Done quando:** una query che descrive un evento breve in una registrazione lunga riesce a far emergere il file e il segmento plausibili, con una valutazione separata.

## P3 — Workflow da sound designer

Implementare solo dopo aver parlato con utenti e scelto una o due priorità:

- [ ] Preferiti / shortlist della sessione.
- [ ] Tag manuali e note locali.
- [ ] Esportazione CSV dei risultati con percorso, score, timestamp (quando disponibile) e metadati.
- [ ] Ricerca per nome/tag tradizionale combinata con semantica.
- [ ] Rilevamento duplicati o file quasi duplicati, con metrica esplicita.

**Done quando:** almeno una feature viene confermata utile da utenti del target e non rompe privacy/cache.

## P4 — Prestazioni e distribuzione

Da attivare solo in base a misure e feedback:

- [ ] Profilare memoria e latenza su 1k, 10k e più file, se disponibili.
- [ ] Valutare FAISS o altro indice vettoriale soltanto quando NumPy diventa un collo di bottiglia misurato.
- [ ] Valutare un wrapper desktop solo dopo aver verificato la domanda; non partire subito con packaging multipiattaforma.
- [ ] Documentare versionamento/migrazione della cache e gestione di file mancanti/rinominati.

## P5 — Validazione commerciale e portfolio

- [ ] Fare 3–5 brevi interviste a sound designer, producer o studi.
- [ ] Chiedere come cercano oggi i sample, quanto spesso perdono tempo, quali workaround usano e cosa renderebbe il tool affidabile.
- [ ] Non chiedere soltanto “ti piace?”: osservare una persona mentre esegue una ricerca reale.
- [ ] Registrare una demo di 60–90 secondi su una libreria legale e realistica.
- [ ] Preparare README in inglese, screenshot, diagramma architetturale e sezione di valutazione.
- [ ] Preparare una candidatura mirata: problema, architettura, test, limiti, decisioni tecniche e risultato misurato.

## Non fare subito

- Non aggiungere BPM o key detection prima di capire se il target usa SoundScout per loop musicali o per sound effects.
- Non aggiungere cloud, login o un database remoto: la proposta attuale è local-first.
- Non usare score di similarità come “percentuale di confidenza”.
- Non ottimizzare per decine di migliaia di file prima di misurare la baseline.
- Non considerare i cinque sample sintetici un dataset valido per dichiarare qualità del retrieval.
