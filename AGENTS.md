# Istruzioni per agenti di coding

Queste regole si applicano a ogni modifica a SoundScout.

## Prima di modificare

1. Leggi `ANTIGRAVITY_START_HERE.md`, `PROJECT_HANDOFF.md`, `ARCHITECTURE.md` e `ROADMAP.md`.
2. Ispeziona il codice esistente: non assumere che documentazione e implementazione coincidano perfettamente.
3. Esegui i test disponibili prima di apportare cambiamenti e riporta l'esito.
4. Identifica la modifica minima che soddisfa la milestone attiva.

## Regole di implementazione

- Non riscrivere tutto da zero e non sostituire lo stack senza una ragione tecnica concreta.
- Mantieni l'app local-first. Non caricare audio o metadati privati su servizi esterni.
- Non introdurre API key, telemetria, login, cloud o analytics senza consenso esplicito del proprietario del progetto.
- Mantieni le responsabilità separate tra `app.py`, `audio.py`, `model.py`, `index.py` e `search.py`.
- Scrivi test per bug fix, logica di ranking, cache e gestione degli errori.
- Evita modifiche distruttive a file o librerie audio. Gli originali devono essere read-only dal punto di vista dell'app.
- Non usare `pickle` per leggere dati arbitrari. Per matrici numeriche usare formati sicuri come NPY con `allow_pickle=False`.
- Valuta gli errori per file in modo robusto: un file corrotto non dovrebbe interrompere necessariamente tutta l'indicizzazione.
- Non fingere di aver eseguito test, scaricato modelli o provato una UI se l'ambiente non lo consente.
- I punteggi di similarità non sono probabilità; non etichettarli come confidenza percentuale.
- Non implementare BPM, key detection, timestamp o funzionalità future senza aggiungere test e aggiornare la documentazione.
- Non aggiungere dipendenze grandi senza descrivere il perché e il costo in installazione.

## Workflow richiesto

1. Enuncia brevemente il problema tecnico e il piano.
2. Fai una modifica circoscritta.
3. Esegui i test pertinenti e aggiungine per i nuovi comportamenti.
4. Aggiorna README/documenti se cambiano istruzioni o limiti.
5. Concludi con: file modificati, comportamento aggiunto, comandi eseguiti, esiti reali dei test, problemi aperti.

Se un test fallisce, non ignorarlo né rimuoverlo semplicemente per rendere verde la suite. Spiega la causa e correggi il problema.
