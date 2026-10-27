# Piano di validazione

## A. Smoke test manuale

1. Attivare l'ambiente virtuale e installare dipendenze.
2. Eseguire `python -m pytest -q`.
3. Avviare `streamlit run app.py`.
4. Se necessario generare la libreria demo con `python scripts/generate_test_library.py ./demo_audio`.
5. Inserire nella UI il percorso assoluto di `demo_audio/` e avviare l'indicizzazione.
6. Provare alcune query: `a sustained pure tone`, `short burst of noise`, `low rumbling sound`, `metallic ringing`, `sweeping whoosh`.
7. Verificare che i risultati siano ascoltabili, i filtri funzionino e la seconda indicizzazione riutilizzi gli embedding.
8. Ripetere il test con una cartella che contenga un formato corrotto o non supportato; controllare i messaggi di errore.

Le query sono ipotesi di prova, non aspettative di risultato garantite: i sample demo sono sintetici e possono non mappare perfettamente al dominio del modello CLAP.

## B. Dataset per retrieval

Creare un manifest con una riga per file, includendo almeno:

- `file_id`
- `relative_path`
- `source_license` / provenienza
- `duration_seconds`
- `notes` (facoltativo)

Creare un file query con almeno:

- `query_id`
- `query_text`
- `relevant_file_ids` (lista di ID)
- `query_type` (ad es. action, material, texture, ambience, distance, space)

Ogni query dovrebbe avere almeno un file rilevante annotato. Se ci sono più risposte legittime, annotarle tutte: non definire automaticamente “sbagliato” un suono alternativo che soddisfa la descrizione.

## C. Metriche

Per ogni query, calcolare la posizione dei file annotati come rilevanti.

- **Recall@5:** numero di file rilevanti recuperati nei primi 5 / numero totale di file rilevanti annotati.
- **Precision@5:** numero di file rilevanti nei primi 5 / 5 (oppure / numero restituito se meno di 5; documentare la convenzione).
- **MRR (facoltativo):** media del reciproco della posizione del primo risultato rilevante.

Riportare la media su query, più alcuni esempi qualitativi di successo e fallimento. Non mescolare query utilizzate per aggiustare il sistema con query tenute da parte per una verifica finale, se il dataset cresce.

## D. Prestazioni e riproducibilità

Annotare:

- sistema operativo e CPU/GPU;
- versioni Python, PyTorch, Transformers e SoundFile;
- checkpoint CLAP usato;
- numero e durata totale dei file;
- tempo primo avvio/download modello;
- tempo di prima indicizzazione e indicizzazione incrementale;
- latenza mediana di query e numero di risultati;
- eventuali errori o file saltati.

## E. Criterio di rilascio portfolio

Non dichiarare “accuratezza alta” senza una definizione e misure. Presentare invece: dataset, metodologia, metriche, limiti, latenza e 2–3 errori analizzati. Anche un risultato mediocre è utile se la valutazione è onesta e porta a una scelta tecnica motivata.
