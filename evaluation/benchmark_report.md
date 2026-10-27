# SoundScout — Retrieval Quality Benchmark Report

> **Valutazione oggettiva della ricerca semantica (Milestone P1)** condotta sul dataset sintetico formalizzato.

## 1. Ambiente di Test e Hardware
- **Sistema Operativo:** Windows-10-10.0.26200-SP0
- **Python:** 3.11.5
- **PyTorch:** 2.12.0+cpu (Device: `cpu`)
- **Transformers:** 4.57.6
- **Checkpoint CLAP:** `laion/clap-htsat-unfused`
- **Dataset:** 36 file audio sintetici in 6 categorie (Impacts, Ambiences, Foley, Tones, Mechanical, Transitions)

## 2. Prestazioni di Indicizzazione e Latenza
| Metrica | Risultato | Note |
|---|---|---|
| File totali indicizzati | **36** | WAV 48 kHz mono / segmenti 10s |
| Tempo prima indicizzazione | **12.23s** | Throughput: **2.9 file/s** su CPU |
| Tempo re-indicizzazione (cache) | **125.9 ms** | 100% riuso vettori invariati |
| Latenza mediana query | **78.4 ms** | Embedding testo + cosine search NumPy |
| Latenza 95° percentile (p95) | **107.2 ms** | Calcolata su 20 query |

## 3. Metriche di Retrieval Complessive
| Metrica | Punteggio | Definizione |
|---|---|---|
| **Recall@1** | **40.00%** | File rilevante recuperato esattamente al primo posto |
| **Recall@3** | **77.50%** | Almeno un file rilevante nei primi 3 risultati |
| **Recall@5** | **85.00%** | Almeno un file rilevante nei primi 5 risultati |
| **Precision@1** | **40.00%** | Precisione sul primo risultato restituito |
| **Precision@5** | **18.00%** | Quota di risultati rilevanti nella top-5 |
| **MRR (Mean Reciprocal Rank)** | **0.6184** | Media dell'inverso della posizione del primo risultato |

## 4. Prestazioni per Categoria di Query
| Categoria Query | N. Query | Recall@1 | Recall@5 | MRR |
|---|---|---|---|---|
| `action` | 1 | 0.00% | 100.00% | 0.2500 |
| `ambience` | 5 | 40.00% | 80.00% | 0.6167 |
| `foley` | 4 | 75.00% | 75.00% | 0.7586 |
| `material` | 5 | 20.00% | 100.00% | 0.5667 |
| `mechanical` | 2 | 100.00% | 100.00% | 1.0000 |
| `tone` | 3 | 0.00% | 66.67% | 0.3889 |

## 5. Analisi Qualitativa: Successi e Casi Limite

### Esempi di Corrispondenza Perfetta (Top-1, MRR = 1.0)
- **Query:** *"sharp glass shatter ping with high resonance"* (`material`)
  - Target atteso: `glass_shatter_ping.wav`
  - Top 3 recuperati: `['IMP_03', 'IMP_01', 'IMP_04']`
- **Query:** *"steady rain downpour texture on hard ground"* (`ambience`)
  - Target atteso: `rain_steady_downpour.wav`
  - Top 3 recuperati: `['AMB_01', 'AMB_05', 'AMB_06']`
- **Query:** *"outdoor night crickets chirping rhythmically"* (`ambience`)
  - Target atteso: `forest_night_crickets.wav`
  - Top 3 recuperati: `['AMB_04', 'TRN_03', 'IMP_06']`
- **Query:** *"metal keys jangling and clinking together"* (`foley`)
  - Target atteso: `keys_jangling_metal.wav`
  - Top 3 recuperati: `['FOL_05', 'AMB_04', 'MEC_02']`
- **Query:** *"slow creaking old wooden door opening"* (`foley`)
  - Target atteso: `door_creak_slow.wav`
  - Top 3 recuperati: `['FOL_03', 'AMB_04', 'TRN_05']`

### Casi di Mancata Corrispondenza / Errore (MRR < 0.33)
- **Query:** *"quiet interior room tone with electrical hum"* (`ambience`)
  - Target atteso: `interior_room_tone.wav` (trovato alla posizione **12**)
  - Top 3 recuperati: `['TRN_03', 'TON_01', 'TRN_05']`
  - *Diagnosi:* Segnale con caratteristiche spettrali affini ma confuso semanticamente dal modello.
- **Query:** *"footsteps walking on loose crunch gravel"* (`foley`)
  - Target atteso: `footsteps_gravel_walk.wav` (trovato alla posizione **29**)
  - Top 3 recuperati: `['AMB_05', 'AMB_01', 'IMP_01']`
  - *Diagnosi:* Segnale con caratteristiche spettrali affini ma confuso semanticamente dal modello.
- **Query:** *"pipe organ church harmonic triad chord"* (`tone`)
  - Target atteso: `church_organ_chord.wav` (trovato alla posizione **6**)
  - Top 3 recuperati: `['FOL_01', 'AMB_05', 'AMB_03']`
  - *Diagnosi:* Segnale con caratteristiche spettrali affini ma confuso semanticamente dal modello.
- **Query:** *"fast air whoosh flyby movement transition"* (`action`)
  - Target atteso: `whoosh_fast_air_pass.wav` (trovato alla posizione **4**)
  - Top 3 recuperati: `['FOL_02', 'FOL_04', 'AMB_02']`
  - *Diagnosi:* Segnale con caratteristiche spettrali affini ma confuso semanticamente dal modello.