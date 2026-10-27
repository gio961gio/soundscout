# Registro iniziale del progetto

## MVP consegnato

- App Streamlit per ricerca testo-audio mediante CLAP.
- Indicizzazione ricorsiva locale.
- Cache incrementale basata su percorso, dimensione e mtime.
- Filtri di metadati, player audio e waveform.
- Cinque file audio sintetici per smoke test.
- Test per segmentazione, filtri, ranking e riuso embedding.
- Documentazione per installazione, architettura e limiti.

## Verifica iniziale registrata

Comando eseguito durante la preparazione del pacchetto:

```bash
python -m pytest -q
```

Risultato: `5 passed`.

La prova completa dell'app Streamlit e il caricamento del checkpoint CLAP devono essere eseguiti sull'ambiente di sviluppo dell'utente, se non già avvenuti. Nessuna valutazione della qualità semantica è stata dichiarata.
