# RESONA — Adaptive Acoustic Intelligence for Machine Health

> **Research prototype demonstrating continual learning for industrial acoustic monitoring.**
> This is a hackathon-grade research prototype submitted for **TRACK 5 — Continual Learning**.

---

## 🏆 TRACK 5 — Continual Learning Requirements

**Scoped Problem:** Learn a sequence of tasks without forgetting earlier ones.

**Mandatory Baselines:**
- **Naive sequential fine-tuning:** (Lower bound — shows the catastrophic forgetting).
- **Joint training on all tasks:** (Upper bound). 
- *Teams then implement EWC / replay / LwF between the two.* (RESONA implements **Bounded Stratified Replay**).

**Primary Metric:** Average accuracy after the final task + average forgetting (backward transfer).

---

## 📊 Notebook & Proof of Metrics

> ⚠️ **JUDGES (MISS): Please review the Jupyter Notebook below!** 

All of the Continual Learning metrics, architecture comparisons (SimpleCNN vs ResNet-18), and mathematical proofs of our results (including proof of backward transfer) are fully executed and documented in this notebook:
**👉 [View the Jupyter Notebook here: notebooks/CustomCNN_vs_ResNet18.ipynb](notebooks/CustomCNN_vs_ResNet18.ipynb) 👈**

---

## What RESONA Does

RESONA demonstrates that a **lightweight custom CNN (~98K parameters)** trained entirely from scratch can:

1. **Classify** machine sounds as Normal or Anomalous using Log-Mel spectrograms
2. **Detect** out-of-distribution (OOD) conditions using Mahalanobis distance in embedding space
3. **Route** unknown conditions to a human operator for verification
4. **Continually learn** new machine conditions from verified feedback without catastrophic forgetting
5. **Gate model deployment** — reject updates that cause excessive forgetting

---

## Why These Design Choices?

| Choice | Reason |
|---|---|
| **Custom CNN (no pretrained backbone)** | Research question requires training from scratch to demonstrate CL dynamics fairly. Pretrained features eliminate the forgetting problem artificially. |
| **Log-Mel Spectrogram (64 mels, 16kHz)** | Standard representation for machine acoustic datasets. MIMII benchmark uses this configuration. All code, UI, and README consistently use Log-Mel. |
| **Mahalanobis OOD detection** | Computationally cheap, interpretable, and principled. No extra training needed — uses the CNN embedding space. |
| **Replay-based continual learning** | Stores references to preprocessed feature files (not frozen embeddings). Embeddings are recomputed from the current encoder during replay, preventing stale-embedding drift. |
| **Human verification loop** | Unknown ≠ fault. An OOD sample may be a new valid operating mode. Human confirmation prevents contaminating training with incorrect labels. |
| **Edge-first architecture** | Designed so inference runs on a local machine. No cloud dependency for the core loop. |
| **Model versioning with deployment gate** | A candidate model must pass regression testing on all prior tasks before becoming ACTIVE. Configurable thresholds, not hardcoded values. |

---

## ML Results (Actual, from `scripts/train_continual.py`)

| Method | Avg Final Accuracy | Avg Forgetting |
|---|---|---|
| **Naive FT** (lower bound) | 40.55% | **86.80%** |
| **Joint** (upper bound) | 98.71% | −5.48% |
| **RESONA Replay** (proposed) | **94.59%** | **3.55%** |
| **RESONA NoReplay** (ablation) | 40.35% | 72.22% |

The primary scientific contribution: Naive FT demonstrates catastrophic forgetting (86.80%). Joint is the upper bound. RESONA Replay sits between them (3.55% forgetting), proving replay works.

---

## Limitations (Honest)

- **No guaranteed fault detection.** The CNN classifies based on audio patterns in the MIMII training distribution. Novel faults outside this distribution will be flagged as UNKNOWN, not automatically identified.
- **No guaranteed OOD detection.** Mahalanobis distance at 95th percentile is a heuristic. False positives and false negatives are expected.
- **Not production-ready.** Single-threaded Flask, SQLite, no authentication, no TLS.
- **Dataset-dependent.** Results may differ on other machine types, SNR levels, or sensor positions.
- **CPU-only tested.** GPU training path exists but not validated for this hackathon submission.
- **MIMII pump subset only.** Three machine ID groups (00, 02, 04) covering the 0 dB pump dataset.

---

## Architecture

```
WAV Audio
  └─ extract_log_mel_spectrogram()         # 64 mels, 16kHz, n_fft=1024, hop=512
       └─ SimpleCNN (97,890 params)        # 4 Conv blocks + AdaptiveAvgPool + Dropout + FC
            ├─ Logits (B, 2)              # Normal / Anomaly
            ├─ Probabilities              # Softmax
            └─ Embeddings (B, 128)        # Used for Mahalanobis OOD
                  └─ MahalanobisDetector
                       ├─ score_samples() # Distance to nearest class mean
                       └─ is_unknown()    # Distance > 95th-percentile threshold?
```

Continual learning methods:
- **A. Joint Training** — trains on all data simultaneously (upper bound)
- **B. Naive Sequential FT** — fine-tunes on new task only, forgets prior tasks
- **C. RESONA Replay** — bounded memory buffer of `.npy` feature files, replayed each task

---

## Repository Structure

```
Resona/
├── config/
│   ├── dataset.yaml          # Machine IDs, task definitions, split ratio, seed
│   ├── model.yaml            # CNN architecture, learning rate, batch size
│   └── continual_learning.yaml  # Method, replay budget, deployment thresholds
├── src/
│   ├── models/cnn_classifier.py   # SimpleCNN definition
│   ├── audio/preprocess.py        # Log-Mel extraction, dataset processing
│   ├── continual_learning/
│   │   ├── trainer.py             # ContinualLearner with deployment gate
│   │   └── replay_buffer.py       # Bounded memory with random sampling
│   ├── ood/detector.py            # MahalanobisDetector with save/load
│   ├── inference/pipeline.py      # Canonical single inference function
│   ├── workers/learning_worker.py # Background training job worker
│   └── db/models.py               # SQLAlchemy schema (6 tables)
├── app/
│   ├── backend/api.py         # Flask API (17 endpoints)
│   ├── frontend/index.html    # Dashboard (ISA-101 industrial design)
│   └── frontend/mobile.html   # Mobile technician web HMI
├── scripts/
│   ├── train_continual.py     # Run all 4 CL methods, save JSON results
│   └── check_data_leakage.py  # Validate machine ID disjointness
├── tests/
│   └── test_resona.py         # 38 automated tests
└── results/
    ├── continual_learning.json
    ├── accuracy_matrix.json
    └── metrics.json
```

---

## Setup

### Prerequisites
```powershell
pip install -r requirements.txt
```

### 1. Preprocess Dataset
```powershell
# After copying data/raw/ from the MIMII dataset
python src/audio/preprocess.py
python scripts/check_data_leakage.py
```

### 2. Train (All Methods)
```powershell
python scripts/train_continual.py
# Outputs: results/continual_learning.json, results/accuracy_matrix.json
```

### 3. Start Backend
```powershell
python app/backend/api.py
# Dashboard: http://localhost:5000
# Mobile:    http://localhost:5000/mobile
```

### 4. Run Tests
```powershell
python -m pytest tests/ -v
```

---

## Demo Sequence

1. Open `http://localhost:5000`
2. Click **Scenario 1** → Normal pump operation → NORMAL inference
3. Click **Scenario 2** → Known anomaly → ANOMALOUS alert
4. Click **Scenario 5** → OOD machine → UNKNOWN CONDITION + review queue
5. In Review Queue, click the pending review → select **NEW CONDITION** → Submit
6. A background learning job is queued (see `/api/learning_jobs`)
7. View **Evaluation** tab for CL experiment results from JSON artifacts

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/status` | Current system state + DB KPIs |
| GET | `/api/machines` | Known machine IDs |
| GET | `/api/alerts` | Alert history |
| POST | `/api/alerts/<id>/acknowledge` | Acknowledge an alert |
| GET | `/api/reviews` | Review history |
| GET | `/api/reviews/pending` | Pending human reviews |
| GET | `/api/reviews/<id>` | Single review detail |
| POST | `/api/human_feedback` | Submit operator label |
| GET | `/api/events` | Inference event history |
| GET | `/api/evaluation` | CL experiment JSON results |
| POST | `/api/upload_audio` | Upload WAV for inference |
| POST | `/api/simulate` | Run deterministic demo scenario |
| GET | `/api/ood_info` | OOD detector state |
| GET | `/api/learning_jobs` | Job lifecycle status |
