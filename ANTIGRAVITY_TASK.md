# Prompt iniziale da incollare in Antigravity

Agisci come senior engineer pragmatico con esperienza in Python, audio DSP e machine learning applicato all'audio. Stai lavorando su **SoundScout**, un MVP esistente: non crearne uno nuovo e non riscriverlo da zero.

## Contesto

Leggi prima questi file, nell'ordine:
1. `ANTIGRAVITY_START_HERE.md`
2. `PROJECT_HANDOFF.md`
3. `ARCHITECTURE.md`
4. `ROADMAP.md`
5. `VALIDATION_PLAN.md`
6. `AGENTS.md`
7. `README.md`

Il prodotto cerca effetti sonori in una libreria locale usando descrizioni testuali. Usa Streamlit per la UI e CLAP (`laion/clap-htsat-unfused`) per embedding audio/testo. Sono già presenti indicizzazione ricorsiva, cache, ranking cosine, filtri, player, waveform, sample sintetici e cinque test unitari.

## Obiettivo generale

Trasformare questa demo in un progetto di portfolio credibile per un tirocinio in Music Engineering/audio software/DSP e verificare se risolve un problema reale per sound designer. La priorità è robustezza e qualità misurata del retrieval, non quantità di feature.

## Primo incarico: milestone P0

1. Ispeziona i file e verifica se il comportamento reale coincide con la documentazione.
2. Esegui `python -m pytest -q` prima di modificare qualsiasi cosa.
3. Verifica ambiente e dipendenze installate; se mancano, comunica in modo preciso i comandi necessari. Non dichiarare l'avvio end-to-end completato se non puoi eseguirlo davvero.
4. Esegui un'analisi statica della UI, del decoder audio, della cache e degli errori.
5. Se l'ambiente lo consente, avvia `streamlit run app.py`, indicizza `demo_audio/` e prova almeno una query. Se non lo consente, indica quale passo è stato bloccato e come l'utente lo può verificare localmente.
6. Cerca bug bloccanti o fragilità concrete e risolvili con modifiche piccole, testate e documentate. Non fare refactor estetici che non migliorano affidabilità o chiarezza.
7. Verifica percorsi mancanti, cartella vuota, file corrotto, cache vuota, cache già presente e re-index incrementale.
8. Concludi con un report in italiano: cosa hai eseguito, risultati, file modificati, problemi residui e prossimo step consigliato.

## Guardrail

- Mantieni local-first: niente upload dei sample, servizi cloud, login, telemetria o chiamate remote aggiuntive.
- Non aggiungere BPM/key detection o segment embedding in questa milestone; sono milestone successive.
- Non chiamare il cosine score “probabilità” o “percentuale di accuratezza”.
- Non modificare o spostare i file audio originali.
- Non aggiungere dipendenze pesanti senza necessità documentata.
- Aggiungi test per ogni bug fix pertinente e mantieni i test esistenti.
- Sii trasparente su ogni comando non eseguito e su ogni verifica che richieda intervento umano.

Inizia ora ispezionando il repository e riportando la baseline dei test prima di modificare il codice.
