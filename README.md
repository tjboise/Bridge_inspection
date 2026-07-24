# BridgeGPT — A Hybrid System for Bridge Inspection

<!-- <img width="600" alt="logo" src="https://github.com/user-attachments/assets/cc95e49a-de1d-48d4-9324-f31715fc4120" /> -->

A multimodal framework that combines **Vision-Language Models (VLMs)** and **Convolutional Neural Networks (CNNs)** for automated reasoning-based bridge inspection.

**Live Demo:** [bridgeinspection.streamlit.app](https://bridgeinspection.streamlit.app)

---

## Overview

Traditional CNN-based inspection tools excel at pixel-level defect detection but lack the semantic reasoning needed to interpret structural context. General-purpose VLMs understand language but cannot produce precise segmentation masks. BridgeGPT bridges this gap with a hierarchical **"Semantic Orchestrator – Specialized Executor"** architecture:

- The **VLM (Gemini)** interprets natural language queries and plans which CNN modules to invoke
- The **CNN backend** executes targeted segmentation with pixel-level precision
- The system answers complex queries like *"How severe is the corrosion on the girder?"* and highlights the relevant region in the image

## Features

- **Natural language querying** — Ask inspection questions in plain English
- **Bridge element segmentation** — Identifies bearing, bracing, deck, floor beam, girder, and pier
- **Rust / corrosion detection** — Localizes defect regions on structural elements
- **Corrosion state estimation** — Classifies regions into CS2 (Fair), CS3 (Poor), CS4 (Severe) per AASHTO/BIRM standards
- **Logical operations** — Handles union, intersection, and single-element queries
- **Conversational interface** — Provides narrative explanations alongside visual outputs

## Architecture

```
User Query (natural language)
        │
        ▼
  Gemini VLM (Semantic Orchestrator)
  · Parses intent → structured JSON plan
  · Decides: element / defect / corrosion state query
        │
   ┌────┴────┐
   ▼         ▼
AECIF-Net   CS Model (DeepLabV3+)
Element     Corrosion State
Segmentation  Estimation
   │         │
   └────┬────┘
        ▼
  Mask Logic Executor
  (union / intersection / single)
        │
        ▼
  Annotated Output + Narrative Response
```

## Models

| Model | Task | Source |
|-------|------|--------|
| **AECIF-Net** (HRNet-based) | Bridge element segmentation (7 classes) | [AECIF-Net](https://github.com/itschenyu/AECIF-Net) |
| **DeepLabV3+ CAA** | Corrosion state classification (CS1–CS4) | [Bianchi et al. 2022](https://github.com/beric7/corrosion_cs_classification) |
| **Gemini 2.5 Flash Lite** | Semantic reasoning & planning | Google Generative AI API |

## Installation

```bash
git clone https://github.com/tjboise/Bridge_inspection.git
cd Bridge_inspection
pip install -r requirements.txt
```

Download model weights and place them in `model_data/`:
- `best_epoch_weights.pth` — AECIF-Net element segmentation weights
- `weights_35.pt` — Corrosion state model weights (auto-downloaded on first run via `gdown`)

## Running Locally

1. Set your Google API key in `.streamlit/secrets.toml`:
   ```toml
   GOOGLE_API_KEY = "your_api_key_here"
   ```

2. Launch the app:
   ```bash
   streamlit run web_app.py
   ```

## Dataset

Structural element and defect annotations are sourced from the open-source steel bridge dataset accompanying [AECIF-Net](https://github.com/itschenyu/AECIF-Net). Corrosion state annotations follow [Bianchi et al. (2022)](https://github.com/beric7/corrosion_cs_classification), labelled in accordance with the Bridge Inspector's Reference Manual (BIRM) and AASHTO guidelines.


