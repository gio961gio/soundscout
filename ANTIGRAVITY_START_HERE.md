# SoundScout — handoff per Antigravity

Benvenuto nel progetto. **Apri questa cartella come workspace in Antigravity** e parti da questo documento.

## Leggi prima questi file

1. `PROJECT_HANDOFF.md` — obiettivo, utente target, decisioni e stato attuale.
2. `ARCHITECTURE.md` — struttura e flusso tecnico del codice esistente.
3. `ROADMAP.md` — prossimi step prioritizzati.
4. `VALIDATION_PLAN.md` — come verificare il progetto e misurare la qualità della ricerca.
5. `AGENTS.md` — regole operative da rispettare quando modifichi il codice.
6. `ANTIGRAVITY_TASK.md` — prompt pronto da assegnare all'agente.

Il progetto applicativo è già nella root: `app.py`, cartella `soundscout/`, test in `tests/` e file demo in `demo_audio/`.

## Stato verificato al momento della consegna

- Test automatici: **5 passed** con `python -m pytest -q` nell'ambiente in cui è stato preparato il pacchetto.
- Sono inclusi il codice dell'MVP, cinque sample sintetici, script per rigenerarli e README originale.
- Non è stato verificato qui un avvio end-to-end della UI Streamlit con il download e il caricamento effettivo del modello CLAP. Va verificato sulla macchina di sviluppo.
- La ricerca semantica su file audio reali non è ancora stata valutata su un dataset annotato. Non fare affermazioni sulla precisione prima di misurarla.

## Primo obiettivo per l'agente

Non riscrivere il progetto. Ispeziona il codice, crea/usa un ambiente Python, installa le dipendenze necessarie, esegui i test e prova l'app su `demo_audio/`. Riporta chiaramente cosa funziona, cosa non funziona e quali passaggi manuali sono necessari. Correggi eventuali problemi bloccanti con modifiche piccole e testate.

Poi passa alla prima milestone della `ROADMAP.md`: validazione end-to-end e baseline di qualità della ricerca.
