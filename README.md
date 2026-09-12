#  RESONA

### **Adaptive Acoustic Intelligence for Machine Monitoring**

<p align="center">

**Listen. Detect. Verify. Learn. Adapt.**

</p>

<p align="center">
  <img src="https://img.shields.io/badge/Deep%20Learning-Continual%20Learning-8B5CF6?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Audio-Machine%20Monitoring-06B6D4?style=for-the-badge" />
  <img src="https://img.shields.io/badge/OOD-Detection-F97316?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Human--in--the--Loop-10B981?style=for-the-badge" />
  <img src="https://img.shields.io/badge/A2A-Agent%20Communication-EC4899?style=for-the-badge" />

</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=flat-square&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/TypeScript-Frontend-3178C6?style=flat-square&logo=typescript&logoColor=white" />
  <img src="https://img.shields.io/badge/TailwindCSS-UI-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white" />
</p>

---

## 🚀 What is RESONA?

**RESONA** is an adaptive acoustic machine-monitoring system that uses **Deep Learning + Out-of-Distribution Detection + Human-in-the-Loop Continual Learning** to detect known machine conditions, identify unfamiliar acoustic conditions, and incrementally learn verified new conditions while minimizing catastrophic forgetting.

Instead of deploying a machine-learning model once and leaving it static, RESONA creates a continuous learning loop:

```text
🎙️ Machine Sound
       ↓
🎚️ Audio Preprocessing
       ↓
📊 Log-Mel Spectrogram
       ↓
🧠 Deep Learning Encoder
       ↓
🔢 Acoustic Embedding
       ↓
🎯 Known-Class Classification
       ↓
🔍 OOD / Unknown Detection
       ↓
🤖 Monitoring Agent
       ↓
      ┌───────────────┬────────────────┐
      ↓               ↓                ↓
   NORMAL          ANOMALY          UNKNOWN
      ↓               ↓                ↓
    LOG             ALERT        HUMAN REVIEW
                                       ↓
                              👤 Verified Feedback
                                       ↓
                              🧠 Learning Agent
                                       ↓
                            ♻️ Continual Learning
                                       ↓
                             📦 Updated Model
                                       ↓
                         🧪 Regression Evaluation
                                       ↓
                         🚀 Deploy / 🔙 Rollback
```

---

# 🎯 Problem

Industrial machines can develop acoustic changes associated with abnormal operating conditions.

However, a deployed model faces a fundamental problem:

> **What happens when a new condition appears that was not present during training?**

A conventional classifier may confidently assign an unfamiliar sound to an existing class.

Even worse, simply fine-tuning the model on the new condition can cause it to **forget previously learned knowledge**.

RESONA addresses this through:

* 🎧 Acoustic Deep Learning
* 🔍 OOD / Unknown Detection
* 👤 Human Verification
* ♻️ Continual Learning
* 🧠 Experience Replay
* 📊 Forgetting Evaluation
* 🤖 Agent-based Orchestration

---

# 💡 Core Idea

### Traditional ML lifecycle

```text
TRAIN → DEPLOY → STATIC MODEL
```

### RESONA lifecycle

```text
TRAIN
  ↓
DEPLOY
  ↓
MONITOR
  ↓
UNKNOWN CONDITION
  ↓
HUMAN VERIFICATION
  ↓
CONTINUAL LEARNING
  ↓
NEW MODEL
  ↓
REGRESSION TEST
  ↓
SAFE DEPLOYMENT
  ↓
CONTINUE LEARNING
```

The central research question is:

> **Can a deep-learning acoustic monitoring model continuously learn new conditions from verified feedback while minimizing catastrophic forgetting of previously learned conditions?**

---

# ✨ Key Features

| Feature                | Description                                        |
| ---------------------- | -------------------------------------------------- |
| 🎧 Acoustic Monitoring | Machine condition monitoring using sound           |
| 🧠 Deep Learning       | Learns acoustic representations automatically      |
| 📊 Log-Mel Features    | Converts audio into time-frequency representations |
| 🔍 OOD Detection       | Identifies unfamiliar conditions                   |
| 👤 Human-in-the-Loop   | Operator verifies unknown cases                    |
| ♻️ Continual Learning  | Learns new tasks sequentially                      |
| 🧠 Experience Replay   | Retains representative previous samples            |
| 📉 Forgetting Analysis | Measures retention of previous knowledge           |
| 🤖 Agent Orchestration | Monitoring, Review and Learning agents             |
| 🔄 Model Versioning    | Supports controlled model updates                  |
| 🛡️ Regression Gate    | Prevents harmful model updates                     |
| 📱 Operator Dashboard  | Real-time monitoring and review interface          |

---

# 🧠 Deep Learning Pipeline

```text
Raw Audio
    │
    ▼
┌───────────────────┐
│ Audio Preprocess  │
│ Resampling        │
│ Normalization     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ STFT              │
│ Time-Frequency    │
│ Representation    │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Mel Filter Bank   │
│ Log Scaling       │
└─────────┬─────────┘
          │
          ▼
┌────────────────────────┐
│ Deep Learning Encoder  │
│ CNN / Audio Transformer│
└──────────┬─────────────┘
           │
           ▼
      🔢 Embedding
           │
      ┌────┴────┐
      ▼         ▼
Classifier     OOD
      │         │
      └────┬────┘
           ▼
   Known / Unknown
```

---

# 🔬 Continual Learning

RESONA evaluates learning across a sequence of tasks rather than treating training as a single static event.

### Task progression

```text
Task 1
  ↓
Task 2
  ↓
Task 3
  ↓
Task N
```

After each task, the model is evaluated on previous tasks.

### Example accuracy matrix

| Model After → | Task 1 | Task 2 | Task 3 |
| ------------- | -----: | -----: | -----: |
| Task 1        |    95% |      — |      — |
| Task 2        |    87% |    93% |      — |
| Task 3        |    84% |    90% |    94% |

This allows us to measure how much previous knowledge was retained.

---

# 🆚 Experimental Baselines

RESONA is evaluated against the required continual-learning references.

### 1️⃣ Naive Sequential Fine-Tuning

```text
Task 1
  ↓
Task 2
  ↓
Task 3
```

The model is sequentially fine-tuned without an explicit forgetting-prevention mechanism.

**Purpose:** Demonstrate the forgetting problem.

---

### 2️⃣ RESONA Continual Learning

```text
New Task
   +
Replay Memory
   ↓
Continual Update
   ↓
Updated Model
```

**Purpose:** Learn new conditions while retaining previous knowledge.

---

### 3️⃣ Joint Training

```text
Task 1 + Task 2 + Task 3
          ↓
     Single Training
```

**Purpose:** Reference / upper-bound comparison because all task data is available simultaneously.

---

# 📈 Evaluation

Primary continual-learning metrics:

### Average Accuracy

Measures final performance across learned tasks.

$$
A_{avg} =
\frac{1}{T}
\sum_{i=1}^{T} A_i
$$

### Average Forgetting

Measures how much performance on previous tasks decreases after learning subsequent tasks.

$$
F_i =
\max_k A_{i,k} - A_{i,T}
$$

where:

* \(A_{i,k}\) = accuracy on task \(i\) after task \(k\)
* \(A_{i,T}\) = final accuracy on task \(i\)

### Goal

```text
             ↑ Average Accuracy
             ↓ Average Forgetting

              RESONA
                 ★
```

---

# 🔍 Unknown / OOD Detection

A standard classifier can be highly confident even for unfamiliar inputs.

Therefore RESONA separates:

```text
KNOWN
 ├── Normal
 └── Known Anomaly

UNKNOWN
 ├── New Condition
 ├── Environmental Shift
 ├── Unseen Machine
 └── Other Out-of-Distribution Input
```

An embedding-based distance / distributional criterion can be used to determine whether an observation sufficiently matches known conditions.

> **Unknown does not automatically mean faulty.**

Unknown means:

> **The system does not have sufficient evidence that the input belongs to its known distribution.**

---

# 👤 Human-in-the-Loop Learning

When an unfamiliar condition is detected:

```text
             UNKNOWN
                ↓
       Human Verification
          /      |       \
         /       |        \
        ↓        ↓         ↓
     NORMAL   KNOWN      NEW
              FAULT    CONDITION
                         ↓
                  Continual Learning
```

Human feedback prevents uncertain predictions from automatically becoming training labels.

---

# 🤖 Agent Architecture

RESONA uses lightweight specialized agents for orchestration.

### 🟢 Monitoring Agent

Responsible for:

* receiving model predictions
* routing normal events
* triggering anomaly alerts
* forwarding unknown cases for verification

---

### 🟡 Human Review Agent

Responsible for:

* presenting unknown cases
* collecting operator feedback
* converting verified feedback into structured learning data

---

### 🔵 Learning Agent

Responsible for:

* validating new samples
* triggering continual learning
* evaluating old and new tasks
* calculating forgetting
* deciding whether the new model is safe to deploy

---

## Agent Communication

Conceptually:

```text
Monitoring Agent
       │
       │ UNKNOWN_DETECTED
       ▼
Human Review Agent
       │
       │ HUMAN_VERIFIED
       ▼
Learning Agent
       │
       │ MODEL_UPDATED
       ▼
Evaluation
       │
       ├──── PASS ────→ 🚀 DEPLOY
       │
       └──── FAIL ────→ 🔙 ROLLBACK
```

A2A-style structured communication can be used as the communication abstraction between these specialized components.

> The agents orchestrate the workflow; the deep-learning model performs acoustic perception.

---

# 🏗️ System Architecture

```text
                         🏭 INDUSTRIAL ENVIRONMENT
                                  │
                             🎙️ Microphone
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Edge Gateway  │
                         └────────┬────────┘
                                  │
                                  ▼
                         🎧 Audio Pipeline
                                  │
                         STFT / Log-Mel
                                  │
                                  ▼
                      ┌──────────────────────┐
                      │ Deep Learning Model  │
                      │ CNN / Transformer    │
                      └──────────┬───────────┘
                                 │
                         ┌───────┴────────┐
                         ▼                ▼
                   Classification       OOD
                         │                │
                         └───────┬────────┘
                                 ▼
                         Monitoring Agent
                                 │
                  ┌──────────────┼──────────────┐
                  ▼              ▼              ▼
               NORMAL         ANOMALY        UNKNOWN
                  │              │              │
                  ▼              ▼              ▼
                 LOG            ALERT       HUMAN REVIEW
                                                │
                                                ▼
                                       VERIFIED FEEDBACK
                                                │
                                                ▼
                                         Learning Agent
                                                │
                                                ▼
                                      Continual Learning
                                                │
                                         Replay Buffer
                                                │
                                                ▼
                                       Updated Model
                                                │
                                                ▼
                                       Regression Tests
                                                │
                                  ┌─────────────┴───────────┐
                                  ▼                         ▼
                              DEPLOY 🚀                 REJECT 🔙
```

---

# 🧰 Technology Stack

## 🧠 Machine Learning

<p>
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Python-ML%20Pipeline-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</p>

* Python
* PyTorch
* TorchAudio / audio processing tools
* NumPy
* SciPy
* Scikit-learn

---

## 🎧 Audio / Signal Processing

<p>
<img src="https://img.shields.io/badge/Audio-STFT-8B5CF6?style=for-the-badge" />
<img src="https://img.shields.io/badge/Features-Log--Mel-06B6D4?style=for-the-badge" />
</p>

* STFT
* Mel Spectrogram
* Log-Mel Spectrogram
* Spectral features
* Audio normalization
* Noise-aware preprocessing

---

## 🧠 Deep Learning Architecture

Depending on the experimental configuration:

* CNN-based acoustic encoder
* CRNN
* Audio Transformer
* Transfer learning
* Learned acoustic embeddings
* Classification head
* OOD detection layer

---

## ♻️ Continual Learning

<p>
<img src="https://img.shields.io/badge/Continual%20Learning-Replay-10B981?style=for-the-badge" />
<img src="https://img.shields.io/badge/Forgetting-Catastrophic%20Forgetting-EF4444?style=for-the-badge" />
</p>

* Sequential task learning
* Experience Replay
* Fixed memory budget
* Task-wise evaluation
* Accuracy matrix
* Average Accuracy
* Average Forgetting
* Backward Transfer
* Ablation studies

---

## 🤖 Agent Layer

<p>
<img src="https://img.shields.io/badge/Agents-Multi--Agent-EC4899?style=for-the-badge" />
<img src="https://img.shields.io/badge/A2A-Agent%20Communication-F97316?style=for-the-badge" />
</p>

* Monitoring Agent
* Human Review Agent
* Learning Agent
* Structured agent messages
* Event-driven orchestration
* Model update lifecycle

---

## ⚡ Backend

<p>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/Uvicorn-Server-499848?style=for-the-badge" />
</p>

* FastAPI
* REST APIs
* WebSocket-based live updates
* Model inference service
* Agent communication endpoints

---

## 🎨 Frontend

<p>
<img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
<img src="https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white" />
<img src="https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white" />
<img src="https://img.shields.io/badge/TailwindCSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white" />
</p>

* React
* TypeScript
* Vite
* Tailwind CSS
* Interactive monitoring dashboard
* Live prediction visualization
* Human-review interface
* Continual-learning progress
* Evaluation dashboard

---

# 📊 Dashboard

### Live Machine Monitoring

```text
┌─────────────────────────────────────────────┐
│             RESONA LIVE MONITOR             │
├─────────────────────────────────────────────┤
│                                             │
│  MACHINE: PUMP-02                           │
│                                             │
│  🎧 Audio Stream                            │
│                                             │
│  📊 Spectrogram                             │
│                                             │
│  STATUS:        🟢 NORMAL                   │
│  ANOMALY:       0.08                        │
│  OOD SCORE:     0.12                        │
│  MODEL:         resona-v1                   │
│                                             │
└─────────────────────────────────────────────┘
```

---

# 🚨 Unknown Condition Workflow

```text
┌─────────────────────────────────────────────┐
│         ⚠️ UNKNOWN CONDITION                │
├─────────────────────────────────────────────┤
│                                             │
│ Machine: PUMP-02                            │
│ OOD Score: 0.91                             │
│ Model: resona-v1                            │
│                                             │
│ What is this condition?                     │
│                                             │
│ [ NORMAL ] [ KNOWN ANOMALY ] [ NEW ]        │
│                                             │
└─────────────────────────────────────────────┘
```

---

# 🧪 Evaluation Dashboard

```text
┌─────────────────────────────────────────────┐
│          CONTINUAL LEARNING                │
├─────────────────────────────────────────────┤
│                                             │
│ Task 1 ████████████████████ 94%             │
│ Task 2 ██████████████████   89%             │
│ Task 3 ███████████████████ 92%             │
│                                             │
│ Average Accuracy     91.6%                  │
│ Average Forgetting    4.2%                  │
│                                             │
│ Naive FT             ███████                │
│ RESONA               █████████████          │
│ Joint Training       ███████████████        │
│                                             │
└─────────────────────────────────────────────┘
```

> Values shown above are illustrative UI examples, not experimental results.

---

# 🗂️ Dataset

## Primary Dataset

### MIMII — Malfunctioning Industrial Machine Investigation and Inspection

Machine categories include:

* 🌀 Fans
* 💧 Pumps
* 🔧 Valves
* 🛤️ Slide Rails

The dataset provides normal and anomalous machine sounds with industrial background noise.

Official dataset:

**MIMII Dataset — Zenodo**

---

# 🔐 Reproducibility

RESONA follows the hackathon reproducibility requirements:

```text
Seed = 42
```

The experiment maintains:

* fixed train/test splits
* deterministic configuration where practical
* no test-data training
* no test-data threshold tuning
* reproducible task sequence
* fixed replay budget
* explicit experiment configuration

---

# 🛡️ Data Leakage Prevention

The evaluation pipeline follows:

```text
TRAIN
 ├── Model training
 ├── Continual adaptation
 └── Replay buffer

VALIDATION
 ├── Threshold selection
 ├── Hyperparameter selection
 └── Model selection

TEST
 └── FINAL EVALUATION ONLY
```

The final test set remains untouched until evaluation.

---

# 🔬 Ablation Study

At least one component is systematically removed or changed.

Example:

### Full Model

```text
Continual Learning
+
Replay Buffer
```

### Ablation

```text
Continual Learning
+
No Replay
```

This allows us to quantify the contribution of replay toward reducing forgetting.

---

# 📦 Project Structure

```text
resona/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── configs/
│   ├── base.yaml
│   ├── experiment.yaml
│   └── continual_learning.yaml
│
├── src/
│   ├── audio/
│   │   ├── preprocessing.py
│   │   ├── spectrogram.py
│   │   └── augmentation.py
│   │
│   ├── models/
│   │   ├── encoder.py
│   │   ├── classifier.py
│   │   └── ood.py
│   │
│   ├── continual/
│   │   ├── replay.py
│   │   ├── trainer.py
│   │   └── metrics.py
│   │
│   ├── agents/
│   │   ├── monitoring.py
│   │   ├── human_review.py
│   │   └── learning.py
│   │
│   └── evaluation/
│       ├── accuracy.py
│       ├── forgetting.py
│       └── ablation.py
│
├── backend/
│   ├── api/
│   ├── websocket/
│   └── inference/
│
├── frontend/
│   ├── components/
│   ├── pages/
│   └── dashboard/
│
├── experiments/
│   ├── naive_ft/
│   ├── resona_replay/
│   └── joint_training/
│
├── checkpoints/
│
├── scripts/
│
├── tests/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# ⚡ Quick Start

```bash
git clone <repository-url>
cd resona

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Run the experiment:

```bash
python scripts/run_experiment.py
```

Run the backend:

```bash
uvicorn backend.main:app --reload
```

Run the frontend:

```bash
cd frontend
npm install
npm run dev
```

---

# 🧪 Experimental Protocol

RESONA compares:

```text
                 SAME TASK SEQUENCE
                        │
          ┌─────────────┼─────────────┐
          ↓             ↓             ↓
       Naive FT      RESONA       Joint Train
          │             │             │
          ↓             ↓             ↓
      Accuracy      Accuracy       Accuracy
      Forgetting    Forgetting     Forgetting
          │             │             │
          └─────────────┼─────────────┘
                        ↓
                  Final Analysis
```

The experiment is designed to answer:

1. Can the model learn new tasks?
2. How much does naive fine-tuning forget?
3. Does replay reduce forgetting?
4. How close does RESONA get to joint training?
5. What happens when replay is removed?
6. Can verified unknown conditions become learned knowledge?

---

# ⚠️ Limitations

RESONA is an **adaptive acoustic monitoring prototype**, not a universal machine-health diagnosis system.

Important limitations include:

* Acoustic sensing cannot observe every physical failure.
* Overheating, smoke, gas leaks and electrical faults may require other sensors.
* Environmental noise can affect acoustic predictions.
* Microphone placement affects performance.
* Machine operating conditions can cause domain shift.
* Rare faults may have limited training data.
* OOD does not automatically mean "machine failure."
* Human feedback can contain labeling errors.
* Continual learning can still experience forgetting.
* Acoustic anomaly detection does not guarantee causal/root-cause diagnosis.

---

# 🌐 Future Architecture

The long-term production architecture can extend RESONA into a multimodal system:

```text
                  🏭 MACHINE
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
    🎧 AUDIO       📳 VIBRATION   🌡️ TEMPERATURE
       │              │              │
       ▼              ▼              ▼
   Audio DL       Vibration DL   Sensor Model
       │              │              │
       └──────────────┼──────────────┘
                      ▼
               🔗 SENSOR FUSION
                      │
                      ▼
              MACHINE HEALTH STATE
                      │
                      ▼
             CONTINUAL LEARNING
```

Potential future modalities:

* Acoustic
* Vibration
* Temperature
* Current
* Pressure
* Vision
* SCADA / telemetry

---

# 🛣️ Roadmap

### ✅ Phase 1 — Research Prototype

* [x] MIMII dataset integration
* [x] Audio preprocessing
* [x] Deep-learning baseline
* [x] OOD detection
* [x] Continual-learning pipeline
* [x] Replay mechanism
* [x] Naive FT baseline
* [x] Joint-training reference
* [x] Accuracy / forgetting metrics

### 🔄 Phase 2 — Adaptive Monitoring

* [ ] Human verification workflow
* [ ] Agent orchestration
* [ ] Model versioning
* [ ] Regression gate
* [ ] Live monitoring dashboard

### 🚀 Phase 3 — Edge Deployment

* [ ] Edge inference benchmarking
* [ ] Quantization
* [ ] ONNX/TensorRT optimization
* [ ] Multi-machine gateway
* [ ] Offline-first operation

### 🌐 Phase 4 — Industrial Scale

* [ ] Multimodal sensor fusion
* [ ] Fleet-level model management
* [ ] Secure model updates
* [ ] Sensor health monitoring
* [ ] Domain adaptation
* [ ] Federated / privacy-preserving learning

---

# 🏆 Why RESONA?

### Traditional Monitoring

> **Detect → Alert**

### RESONA

> **Detect → Understand Uncertainty → Verify → Learn → Evaluate → Adapt**

The key idea is not simply making another machine-sound classifier.

It is building a system where:

> **The deployed model can evolve as the environment evolves, while explicitly measuring whether previously learned knowledge is being retained.**

---

# 📚 Research Foundation

RESONA builds upon established research areas rather than claiming novelty for individual components:

* Industrial acoustic anomaly detection
* Deep audio representation learning
* Out-of-distribution detection
* Continual learning
* Experience replay
* Human-in-the-loop machine learning
* Agent-to-agent communication
* Edge-oriented inference

All datasets, pretrained models, libraries and reference implementations should be properly cited in the accompanying report.

---

# 👨‍💻 Built For

### Deep Learning Hackathon — Track 5

**Continual Learning**

> *Learn new conditions. Remember the old ones.*

---

<p align="center">

### 🎧 RESONA

**Hear → Detect → Verify → Learn → Remember**

</p>
