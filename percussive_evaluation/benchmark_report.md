# SoundScout — Retrieval Quality Benchmark Report

> **Valutazione oggettiva della ricerca semantica** condotta su: `Freesound One-Shot Percussive Sounds Dataset (UPF MTG / Zenodo 3665275)`.

## 1. Ambiente di Test e Hardware
- **Sistema Operativo:** Windows-10-10.0.26200-SP0
- **Python:** 3.11.5
- **PyTorch:** 2.12.0+cpu (Device: `cpu`)
- **Transformers:** 4.57.6
- **Checkpoint CLAP:** `laion/clap-htsat-unfused`
- **Dataset:** 40 file audio (Freesound One-Shot Percussive Sounds Dataset (UPF MTG / Zenodo 3665275))

## 2. Prestazioni di Indicizzazione e Latenza
| Metrica | Risultato | Note |
|---|---|---|
| File totali indicizzati | **40** | WAV 48 kHz mono / segmenti 10s |
| Tempo prima indicizzazione | **16.60s** | Throughput: **2.4 file/s** su CPU |
| Tempo re-indicizzazione (cache) | **127.4 ms** | 100% riuso vettori invariati |
| Latenza mediana query | **78.8 ms** | Embedding testo + cosine search NumPy |
| Latenza 95° percentile (p95) | **283.2 ms** | Calcolata su 10 query |

## 3. Metriche di Retrieval Complessive
| Metrica | Punteggio | Definizione |
|---|---|---|
| **Recall@1** | **12.50%** | Quota media di target rilevanti recuperati al rank 1 (limitata superiormente da 1/|R|) |
| **Recall@3** | **30.00%** | Quota media di target rilevanti recuperati nella top-3 |
| **Recall@5** | **47.50%** | Quota media di target rilevanti recuperati nella top-5 |
| **Precision@1 (Hit@1)** | **100.00%** | Percentuale di query con risultato al rank 1 rilevante (Top-1 Hit Rate) |
| **Precision@5** | **76.00%** | Quota media di risultati rilevanti nella top-5 |
| **MRR (Mean Reciprocal Rank)** | **1.0000** | Media dell'inverso della posizione del primo risultato rilevante |

## 4. Prestazioni per Categoria di Query
| Categoria Query | N. Query | Recall@1 | Recall@5 | MRR |
|---|---|---|---|---|
| `timbral_producer` | 5 | 12.50% | 52.50% | 1.0000 |
| `transient_drum` | 5 | 12.50% | 42.50% | 1.0000 |

## 5. Analisi Qualitativa: Successi e Casi Limite

### Esempi di Corrispondenza Perfetta (Top-1, MRR = 1.0)
- **Query:** *"punchy acoustic kick drum transient with deep low end"* (`transient_drum`)
  - Target atteso: `102784_acoustic_kick1.wav`
  - Top 3 recuperati: `['PERC_05', 'PERC_07', 'PERC_08']`
- **Query:** *"crisp acoustic snare drum crack with snappy wire attack"* (`transient_drum`)
  - Target atteso: `100190_Rick_DRUM_SNARE_soft.wav`
  - Top 3 recuperati: `['PERC_10', 'PERC_17', 'PERC_15']`
- **Query:** *"metallic closed hi-hat cymbal tick transient"* (`transient_drum`)
  - Target atteso: `102780_acoustic_closed_hihat1.wav`
  - Top 3 recuperati: `['PERC_21', 'PERC_22', 'PERC_23']`
- **Query:** *"dry hand clap percussion snap one-shot"* (`transient_drum`)
  - Target atteso: `105337_so right_Clap.wav`
  - Top 3 recuperati: `['PERC_26', 'PERC_32', 'PERC_21']`
- **Query:** *"bright metallic crash cymbal ring and wash"* (`transient_drum`)
  - Target atteso: `121396_finger_cymbals_crash_choke01.wav`
  - Top 3 recuperati: `['PERC_37', 'PERC_20', 'PERC_19']`

*Nessun fallimento critico riscontrato nella suite di test.*