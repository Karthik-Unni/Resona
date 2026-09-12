# RESONA: Adaptive Acoustic Intelligence for Machine Monitoring

## What is RESONA?
RESONA is a deep-learning system that listens to industrial machine sounds, detects known and unfamiliar machine conditions, uses an agentic workflow to coordinate alerts and human verification, and uses continual learning to incorporate verified new conditions while minimizing forgetting of previously learned knowledge.

## Why machine sound?
Acoustic monitoring provides non-invasive, continuous diagnostic capability for industrial machinery. Unlike vibration sensors which require physical contact, microphones can monitor equipment from a distance and capture a wide range of mechanical and fluid anomalies.

## Dataset
This project uses the official **MIMII Dataset (Public 1.0)**.
- **Source**: [https://zenodo.org/records/3384388](https://zenodo.org/records/3384388)
- **Subset Used**: 0 dB Pump subset (Models 00, 02, 04, and 06).
- **Justification**: A precise subset was chosen to support rigorous continual learning experiments without requiring an unnecessarily large download. Models 00, 02, and 04 are used sequentially for continual learning tasks. Model 06 is held out strictly for the Unknown/OOD condition evaluation.

## Architecture
1. **Audio Preprocessing**: Raw audio → Log-Mel Spectrogram (64 mels).
2. **Deep Learning Audio Encoder**: Lightweight Convolutional Neural Network (CNN).
3. **Unknown-condition Detection**: Mahalanobis distance calculated over the CNN embedding space, calibrated to the 95th percentile of the training set.

## Continual Learning
The system implements **Experience Replay** to prevent catastrophic forgetting.
- A fixed memory budget is maintained per task/class.
- When learning a new condition, a subset of past experiences is interleaved with the new training data.
- Baselines provided: Naive Sequential Fine-Tuning (lower bound) and Joint Training (upper bound).

## Agent Workflow
- **Monitoring Agent**: Evaluates predictions. Logs normal behavior, alerts on known anomalies, and requests human verification for unknown conditions.
- **Human Review Agent**: Facilitates operator feedback (via desktop/mobile QR dashboard) to label unfamiliar conditions.
- **Learning Agent**: Coordinates the continual learning process, validates data, adapts the model, and calculates forgetting metrics before deployment.

## Unknown-condition detection
Instead of blindly forcing unfamiliar sounds into a known class (a common failure mode in softmax-based classification), RESONA measures the Mahalanobis distance in the feature space. Distances exceeding the training-calibrated threshold trigger the Human Review workflow.

## Demo
The demo provides a dynamic dashboard mimicking an edge deployment.
- It includes 9 deterministic test cases: Normal, Known Anomaly, Noisy variants, Unknown Condition, Human Verification, Post-Learning Recognition, Forgetting Evaluation, and Baseline Comparisons.
- A **QR Code** is provided to simulate phone/mobile notification workflows.

## Reproduction
1. Install requirements: `pip install -r requirements.txt`
2. Download dataset: `python scripts/download_mimii.py`
3. Preprocess audio: `python src/audio/preprocess.py`
4. Run Continual Learning: `python scripts/train_continual.py`
5. Verify Data Leakage: `python scripts/check_data_leakage.py`
6. Run Dashboard: `python app/backend/api.py` then open `app/frontend/index.html` in a browser.

## Results
(Results are automatically written to `results/results_RESONA_Replay.txt` during `train_continual.py`).
- **Final Avg Accuracy**: Computed across all learned tasks at the end of the sequence.
- **Avg Forgetting**: Measures backward transfer (how much accuracy was lost on old tasks).

## Limitations
- **Data Availability**: The system is trained on a specific subset of MIMII pump sounds at 0 dB SNR. Performance may vary on drastically different machine types.
- **Hardware Profile**: Inference is currently CPU-bound in the demo. Real-time edge deployment would require further quantization and ONNX/TensorRT optimization.
- **Predictive Maintenance**: The current dataset supports condition monitoring (anomaly detection), not long-term predictive degradation forecasting.

## Citation
MIMII Dataset: Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” arXiv preprint arXiv:1909.09347, 2019.
