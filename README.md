# Nepal-CXR-NET

**A deep learning triage system for lung malignancy detection in TB-endemic regions**

Sorasak Joshi · UCI MSBA Graduate · March 2026

> **Research prototype — not validated for clinical use.** This system has not been evaluated on Nepalese patient data, has not undergone ethics review, and is not approved for deployment in any clinical setting. See [Known Limitations](#known-limitations) for full disclosure.

---

## The Problem

In Nepal, there are roughly 0.7 radiologists per 100,000 people and a TB burden of ~117 cases per 100,000 per year. District GPs read most chest X-rays alone, with no specialist and no second opinion. Because TB is so prevalent, every suspicious nodule is naturally assumed infectious — which means early lung cancer can go undetected in exactly the settings where detection matters most.

Early-stage lung cancer and tuberculosis look nearly identical on a 2D chest radiograph. Systems that treat them as separate classification problems don't address the actual diagnostic challenge. Nepal-CXR-NET models both conditions simultaneously and is designed from the start for the infrastructure constraints of rural Nepal: X-ray only, lightweight models, offline-capable inference.

Full project write-up: [Nepal-CXR-NET on Notion](https://www.notion.so/32e2aebf771881ad8fd6db31c210790b)

---

## What This System Does

Nepal-CXR-NET runs chest X-rays through three components in sequence:

1. **DLBS (Deep Learning Bone Suppression)** — A U-Net-style GAN with MobileNet inverted residual blocks that suppresses rib shadows to reveal more of the lung field. Trained on Gaussian-blur pseudo-pairs (σ=3.0) rather than real bone suppression ground truth.

2. **Dual-Stream Classifier** — DenseNet121 (global context stream) + EfficientNet-B0 (nodule morphology stream), fused with CBAM channel-spatial attention. 4-class sigmoid output: Normal / TB / Malignancy / Uncertain. ~14.77M parameters. Safety Gate down-weights malignancy probability when TB is strongly detected.

3. **YOLOv8-Nano Detector** — 8 pathology classes, 640×640 input, trained on NIH ChestX-ray14 with pseudo-bounding-box labels (not radiologist-annotated).

4. **Triage Scorer** — Assigns Red / Yellow / Green risk levels for clinical routing.

---

## Actual Results (Honest)

| Component | Metric | Value | Caveat |
|-----------|--------|-------|--------|
| TB Classifier | Weighted F1 | 1.000 | 2-class validation only (Normal + TB). Malignancy class absent from validation cache. |
| TB Classifier | TB Recall | 1.000 | Same caveat. |
| TB Classifier | Training stopped | Epoch 25/50 | Early stopping, patience=10 |
| YOLO Detector | mAP@0.50 (all 8 classes) | 0.164 | Pseudo-label training data |
| YOLO Detector | mAP@0.50 (nodule class) | 0.096 | Weak. Attributed to pseudo-labels, not radiologist annotation. |
| DLBS | Validation L1 Loss | 0.0032 | Gaussian blur pseudo-pairs, not real bone suppression ground truth. |
| End-to-end smoke test | Inference (MacBook MPS) | ~2s/image | Single image, not batch |

The four-class balanced evaluation (Normal / TB / Malignancy / Uncertain) has **not been completed**. This is the most important remaining gap.

---

## Known Limitations

1. **No four-class validation.** The malignancy and uncertain classes were absent from the validation cache. F1=1.000 is a two-class result. The system has not been evaluated on its ability to distinguish cancer from TB.

2. **Safety Gate risk.** The Safety Gate suppresses malignancy probability when TB is detected. In a high-TB-prevalence setting like Nepal, this could systematically underweight cancer signals. Configurable soft suppression is implemented but the failure mode is documented and unresolved.

3. **Dual-stream training gap.** Both classifier streams received the same input tensor during training (DLBS output was not connected to classifier input during training runs). The dual-stream benefit is present at inference but the streams were not trained on distinct inputs.

4. **Detector is weak.** mAP@0.50 of 0.096 on nodules is at or near the threshold of practical utility. Retraining on radiologist-annotated bounding boxes is required.

5. **No Nepalese patient data.** All training data is from publicly available academic datasets (see [DATASETS.md](DATASETS.md)). The system has never been evaluated on Nepalese patient radiographs.

6. **Not clinically validated.** No IRB/ethics review. No clinical partnership. Not deployable.

---

## Project Structure

```
nepal_cxr_net/             # Main Python package
├── models/
│   ├── dlbs/              # Bone suppression GAN (generator, discriminator)
│   ├── classifier/        # Dual-stream classifier (context stream, nodule stream, fusion)
│   ├── detector/          # YOLOv8 wrapper
│   └── fusion/            # CBAM attention blocks
├── data/
│   ├── loaders/           # Dataset loaders (TB, NIH, CheXpert, etc.)
│   └── preprocessing/     # CLAHE, normalization, harmonization pipeline
├── training/
│   ├── losses/            # Weighted focal loss, adversarial loss
│   ├── trainers/          # Training loops (classifier, detector, DLBS)
│   └── validators/        # Recall-first validators, comprehensive validator
├── deployment/
│   ├── inference/         # InferencePipeline (preprocess → DLBS → classify → detect → triage)
│   ├── ui/                # Flask app + API key auth
│   ├── cloud/             # ONNX export, cloud inference (planned)
│   ├── edge/              # Quantization (planned)
│   └── jetson/            # TensorRT helpers (planned, not tested)
├── federated/             # FedAvg/FedProx stubs + NVIDIA FLARE scaffolding (research stub only)
├── configs/               # deployment_config.yaml, training_config.yaml
└── utils/                 # Config loader, metrics, visualization overlays

frontend/                  # React 18 + Vite 5 clinical UI
├── src/
│   ├── App.jsx            # Main app: studies list, image viewer, results, report modal
│   └── components/        # ImageViewer, ResultsPanel, ReportModal, etc.
└── vite.config.js

scripts/                   # Training scripts
├── train_tb_classifier.py
├── train_classifier.py
├── train_dlbs.py
├── train_detector.py
└── prepare_nih_bbox.py    # Pseudo-bbox generation for YOLO training

config.yaml                # Root training config (TB/cancer classifier)
config_cancer.yaml         # Cancer-specific training config
```

---

## Installation

```bash
# Python dependencies (requires Python 3.9+)
pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

**Hardware used for training:** Apple Silicon MacBook (MPS backend), batch size 64, 224×224 input.

**Note on model weights:** Model weights are not included in this repo (too large for GitHub). See [Model Weights](#model-weights) below.

---

## Running the Backend

```bash
# From repo root
export CONFIG_PATH="$(pwd)/nepal_cxr_net/configs/deployment_config.yaml"
python -m nepal_cxr_net.deployment.ui.app
```

The Flask API starts on port 8080. Key routes:
- `POST /api/infer` — multipart `image` field, returns JSON with triage, probabilities, optional annotated base64 image
- `GET /api/health` — no auth required
- `POST /api/generate-key` — creates API key (treat as admin in any deployment)

Protected routes require `X-API-Key` header (or `Authorization: Bearer <key>`).

## Running the Frontend

```bash
cd frontend && npm run dev
```

Opens at http://localhost:3000. Proxies `/api` → `http://localhost:8080`. Handles DICOM files, bone suppression overlay, AI result display, and PDF report export.

---

## Model Weights

Model weights are excluded from this repository. If you are a researcher interested in the pretrained weights for evaluation purposes, contact me at sorasakjoshi2@gmail.com with a brief description of your intended use.

**What's available on request:**
- `best_tb_classifier.pth` — DualStreamClassifier trained on TB + Normal classes (~169MB)
- `checkpoints/detector/nih_cxr/weights/best.pt` — YOLOv8-Nano trained on NIH ChestX-ray14 pseudo-labels (~6MB)

**DLBS weights** are not released (model was trained on Gaussian-blur pseudo-pairs and is not a validated bone suppression system).

---

## Datasets Used

See [DATASETS.md](DATASETS.md) for a complete honest account of which datasets were actually used in training vs. which appear in the codebase as planned integrations.

Short version: TB training used public TB CXR datasets (CC BY licensed). YOLO used NIH ChestX-ray14. CheXpert was present for one smoke test image. VinDr-CXR, TBX11K, and JSRT were never downloaded or used.

---

## What's Implemented vs. Aspirational

| Component | Status |
|-----------|--------|
| DLBS GAN | Trained, runs |
| Dual-stream classifier | Trained, runs (training gap noted above) |
| YOLOv8-Nano detector | Trained, runs (weak performance) |
| Triage scorer | Implemented |
| Flask API + auth | Implemented |
| React frontend (DICOM + overlays + PDF) | Implemented |
| ONNX export | Planned, not completed |
| Jetson/TensorRT deployment | Scaffolded, not tested on hardware |
| Federated learning (FLARE/FedProx) | Research stub only |
| Four-class balanced evaluation | Not completed |
| Clinical validation | Not started |

---

## Path Forward

The most important remaining steps are not technical — they're clinical. Four-class validation on a balanced dataset requires radiologist-labeled data from Nepal. That requires a clinical partnership, which requires ethics approval and institutional agreement.

**Near-term goals:**
1. Establish one clinical partnership (Nepalese GP or teaching hospital)
2. Submit ethics application for patient data access
3. Run four-class balanced evaluation
4. Conduct demographic and fairness analysis on the cancer detector
5. Co-design the clinical UI with actual GPs before it gets near a clinical setting

If you are a clinician, researcher, or healthcare organization working in Nepal or similar TB-endemic settings and are interested in collaborating, please reach out.

---

## Contact

**Sorasak Joshi** | UCI MSBA Graduate | March 2026
sorasakjoshi2@gmail.com

---

## License

Code in this repository is released under the MIT License. Model weights are available for academic research use only  contact before use. Training data is subject to the licenses of the respective source datasets (see [DATASETS.md](DATASETS.md)).

---

## Disclaimer

This software is a research prototype. It has not been evaluated in a clinical setting, has not received regulatory clearance, and is not intended for clinical use. The authors make no claims about diagnostic accuracy on real patient populations. Do not use this system to make or inform clinical decisions.
