# SoundScout — Retrieval Quality Benchmark Report

> **Valutazione oggettiva della ricerca semantica** condotta su: `ESC-50 (GitHub) & Clotho DCASE (Tampere University / Zenodo)`.

## 1. Ambiente di Test e Hardware
- **Sistema Operativo:** Windows-10-10.0.26200-SP0
- **Python:** 3.11.5
- **PyTorch:** 2.12.0+cpu (Device: `cpu`)
- **Transformers:** 4.57.6
- **Checkpoint CLAP:** `laion/clap-htsat-unfused`
- **Dataset:** 50 file audio (ESC-50 (GitHub) & Clotho DCASE (Tampere University / Zenodo))

## 2. Prestazioni di Indicizzazione e Latenza
| Metrica | Risultato | Note |
|---|---|---|
| File totali indicizzati | **50** | WAV 48 kHz mono / segmenti 10s |
| Tempo prima indicizzazione | **42.65s** | Throughput: **1.2 file/s** su CPU |
| Tempo re-indicizzazione (cache) | **144.9 ms** | 100% riuso vettori invariati |
| Latenza mediana query | **52.8 ms** | Embedding testo + cosine search NumPy |
| Latenza 95° percentile (p95) | **96.3 ms** | Calcolata su 30 query |

## 3. Metriche di Retrieval Complessive
| Metrica | Punteggio | Definizione |
|---|---|---|
| **Recall@1** | **65.56%** | Quota media di target rilevanti recuperati al rank 1 (max teorico 33.3% per query con 3 target) |
| **Recall@3** | **96.67%** | Almeno un file rilevante nei primi 3 risultati |
| **Recall@5** | **98.89%** | Almeno un file rilevante nei primi 5 risultati |
| **Precision@1 (Hit@1)** | **83.33%** | Percentuale di query con risultato al rank 1 rilevante (Top-1 Hit Rate) |
| **Precision@5** | **32.67%** | Quota media di risultati rilevanti nella top-5 |
| **MRR (Mean Reciprocal Rank)** | **0.9167** | Media dell'inverso della posizione del primo risultato rilevante |

## 4. Prestazioni per Categoria di Query
| Categoria Query | N. Query | Recall@1 | Recall@5 | MRR |
|---|---|---|---|---|
| `environmental` | 10 | 26.67% | 96.67% | 0.9000 |
| `human_caption` | 20 | 85.00% | 100.00% | 0.9250 |

## 5. Analisi Qualitativa: Successi e Casi Limite

### Esempi di Corrispondenza Perfetta (Top-1, MRR = 1.0)
- **Query:** *"shattering glass breaking into sharp pieces"* (`environmental`)
  - Target atteso: `1-20133-A-39.wav`
  - Top 3 recuperati: `['ESC_02', 'ESC_01', 'ESC_03']`
- **Query:** *"human footsteps walking and stepping on hard floor"* (`environmental`)
  - Target atteso: `1-155858-A-25.wav`
  - Top 3 recuperati: `['ESC_06', 'ESC_04', 'ESC_05']`
- **Query:** *"knocking loudly on a solid wooden door"* (`environmental`)
  - Target atteso: `1-101336-A-30.wav`
  - Top 3 recuperati: `['ESC_07', 'ESC_09', 'ESC_08']`
- **Query:** *"blowing howling gusts of cold wind"* (`environmental`)
  - Target atteso: `1-137296-A-16.wav`
  - Top 3 recuperati: `['ESC_16', 'ESC_18', 'CLO_17']`
- **Query:** *"analog clock ticking rhythmic mechanism"* (`environmental`)
  - Target atteso: `1-21934-A-38.wav`
  - Top 3 recuperati: `['ESC_20', 'ESC_21', 'ESC_19']`

*Nessun fallimento critico riscontrato nella suite di test.*