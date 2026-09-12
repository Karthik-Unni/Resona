# RESONA — Adaptive Acoustic Intelligence for Machine Health

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

> **Research prototype demonstrating Continual Learning for industrial acoustic monitoring.**
> This is a hackathon-grade research prototype submitted for **TRACK 5 — Continual Learning**.

---
RESONA — Adaptive Acoustic Intelligence for Machine Monitoring is a deep-learning-based acoustic condition monitoring system designed to detect abnormal machine behavior from sound while continuously adapting to new operating conditions.

The system processes machine audio into Log-Mel spectrograms and uses a lightweight custom CNN to learn acoustic patterns associated with machine conditions. Instead of forcing the classifier to make a prediction when it encounters an unfamiliar sound, RESONA uses Out-of-Distribution (OOD) detection to distinguish between known conditions and previously unseen acoustic patterns.

When an unfamiliar condition is detected, the system creates a human-review request. A technician can verify the condition through the web or mobile interface, and the verified sample is added to a replay buffer. The continual-learning pipeline then uses the new information together with previously learned samples, allowing the model to adapt while reducing catastrophic forgetting.

The project also includes an edge-oriented deployment architecture, where a simulated Raspberry Pi edge node performs local audio preprocessing, inference, and OOD evaluation before sending compact events to the backend. A web-based industrial HMI provides live monitoring, alerts, review workflows, model status, and evaluation results, while a mobile interface enables field technicians to verify detected conditions.

A major focus of the project is not only classification accuracy, but also continual-learning behavior. We compare different learning strategies and analyze metrics such as final accuracy, forgetting, and backward transfer to evaluate whether the system can learn new acoustic conditions while retaining — and in some cases improving — knowledge of previously learned tasks.
## 🏆 TRACK 5 — Continual Learning Requirements

**Scoped Problem:** Learn a sequence of tasks without forgetting earlier ones.

**Mandatory Baselines:**
- **Naive sequential fine-tuning:** (Lower bound — shows catastrophic forgetting).
- **Joint training on all tasks:** (Upper bound). 
- *Teams then implement EWC / replay / LwF between the two.* (RESONA implements **Bounded Stratified Replay**).

**Primary Metric:** Average accuracy after the final task + average forgetting (backward transfer).

---

## 📊 Notebook & Proof of Metrics

> ⚠️ **MISS: Please review the Jupyter Notebook below!** 

All of the Continual Learning metrics, architecture comparisons (SimpleCNN vs ResNet-18), and mathematical proofs of our results (including proof of backward transfer) are fully executed and documented in this notebook:
**👉 [View the Jupyter Notebook here: notebooks/CustomCNN_vs_ResNet18.ipynb](notebooks/CustomCNN_vs_ResNet18.ipynb) 👈**
<img width="862" height="574" alt="image" src="https://github.com/user-attachments/assets/0965ecd8-fc7d-4fd8-af9a-b60ad96e9841" />
Accuracy Improvement Demonstrating Backward Transfer in Continual Learning


---

## 🧠 Deep Learning Architecture & Models Used

RESONA is built from the ground up using a custom Deep Learning pipeline designed specifically for edge-oriented acoustic machine health monitoring. 

### 1. The Core Acoustic Model: Custom CNN
Instead of relying on bloated pretrained visual models like ResNet-18 or VGG (which contain millions of parameters and are prone to overfitting on narrow audio domains), RESONA utilizes a **Custom Lightweight CNN trained entirely from scratch**.
* **Parameters:** ~97,890 (Extremely lightweight, 100x smaller than ResNet-18).
* **Input:** 64-bin Log-Mel Spectrograms extracted from 16kHz audio using a 1024-point FFT and 512 hop length.
* **Architecture:** 4 Convolutional blocks featuring Batch Normalization, ReLU activations, and Max Pooling, followed by Adaptive Average Pooling to handle variable time dimensions, and a Dropout-regularized Fully Connected classifier.
* **Latency:** ~14ms per sample on a standard CPU.

### 2. The Continual Learning Engine: Bounded Stratified Replay
To combat catastrophic forgetting when learning new machine states, RESONA utilizes an active **Replay Buffer**:
* **Mechanism:** When a new acoustic condition is verified by a human, the Continual Learner retrieves a bounded, stratified sample of historical dataset features (`.npy` Log-Mel arrays) representing previously learned tasks.
* **Dynamic Replay:** By replaying raw Log-Mel features rather than frozen latent embeddings, the CNN's feature extractor is allowed to organically adapt its convolutional filters to the new task without its weights drifting away from the representation required for old tasks.

### 3. Out-of-Distribution (OOD) Detection: Mahalanobis Distance
Softmax probabilities are mathematically bounded and notoriously overconfident on unseen data. RESONA completely bypasses Softmax for anomaly detection.
* **Latent Space Modeling:** We extract the 128-dimensional embedding from the penultimate layer of the CNN.
* **Geometric Uncertainty:** We fit class centroids and a shared covariance matrix to the training data representations.
* **Detection:** During inference, we calculate the Mahalanobis statistical distance of the new sample to the known centroids. If the distance exceeds the 95th-percentile calibration threshold, the signal is flagged as **UNKNOWN** (OOD) and routed to a human operator, preventing silent failures.

---

## 🚀 ML Results (Actual, from `scripts/train_continual.py`)

| Method | Avg Final Accuracy | Avg Forgetting |
|---|---|---|
| **Naive FT** (lower bound) | 88.31% | **-2.43%** |
| **Joint** (upper bound) | 89.23% | 0.88% |
| **RESONA Replay** (proposed) | **78.02%** | **20.92%** |
| **RESONA NoReplay** (ablation) | 0.00% (Rejected) | 74.35% |

*Note: Naive FT experienced "backward transfer" (negative forgetting) due to the extreme acoustic similarity of the MIMII pump dataset tasks, where learning Task 3 slightly improved the filters for Task 1.*

---

## 🛠️ Repository Structure

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
│   └── test_resona.py         # 40 automated tests
└── results/
    ├── continual_learning.json
    ├── accuracy_matrix.json
    └── metrics.json
```

---

## 💻 Setup & Execution

### Prerequisites
```powershell
pip install -r requirements.txt
```

### 1. Preprocess Dataset
```powershell
python src/audio/preprocess.py
python scripts/check_data_leakage.py
```

### 2. Train Continual Learning Baseline
```powershell
python scripts/train_continual.py
```

### 3. Start Backend & Dashboard
```powershell
python app/backend/api.py
# Dashboard: http://localhost:5000
```

### 4. Run Automated Tests
```powershell
python -m pytest tests/ -v
```

---

## 🎮 Demo Sequence

1. Open `http://localhost:5000`
2. Click **Scenario 1** → Normal pump operation → NORMAL inference
3. Click **Scenario 2** → Known anomaly → ANOMALOUS alert
4. Click **Scenario 5** → OOD machine → UNKNOWN CONDITION + review queue
5. In Review Queue, click the pending review → select **NEW CONDITION** → Submit
6. A background learning job is queued (see `/api/learning_jobs`)
7. View **Evaluation** tab for CL experiment results from JSON artifacts

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/6706f85d-efc5-4274-848d-b29961e72a35" />
<img width="500" height="900" alt="image" src="https://github.com/user-attachments/assets/f9e5dad0-bbbc-49b1-a046-1c0e66527a17" />
<img width="500" height="900" alt="image" src="https://github.com/user-attachments/assets/75353397-1455-4861-b80c-943a7bcda4ff" />





---

## 🛑 Limitations (Honest)

- **No guaranteed fault detection.** The CNN classifies based on audio patterns in the MIMII training distribution. Novel faults outside this distribution will be flagged as UNKNOWN, not automatically identified.
- **No guaranteed OOD detection.** Mahalanobis distance at 95th percentile is a heuristic. False positives and false negatives are expected.
- **Replay Interference.** The strict 100-sample bounded replay buffer struggled to completely retain Task 2 while learning Task 3, causing a 20% average forgetting spike in our experiment.
