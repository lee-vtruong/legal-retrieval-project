# ViMELD: Vietnamese Multi-view Enhanced Legal Decoupling Framework

This repository contains the official implementation of the paper:  
**"ViMELD: Vietnamese Multi-view Enhanced Legal Decoupling Framework for Legal Document Retrieval"** (ALQAC 2025).

---

## 🚀 Overview

Vietnamese Legal Information Retrieval presents unique challenges due to the dense, non-linear topology of cross-references and the implicit semantics of legal provisions.  

**ViMELD** tackles these challenges by introducing a *"decoupled intelligence"* philosophy, orchestrated in three specialized stages:

### 1. Phase 1: Multi-View Hybrid Retrieval
- Integrates:
	- Sparse (**BM25L**)
	- Dense (**BGE-M3**)
	- Graph Attention Networks (**GAT**)  
- Captures article-level citation topology

### 2. Phase 2: Uncertainty-Aware Meta-Learner
- XGBoost ranker  
- Trained on:
	- Structural features  
	- Semantic features  
	- Uncertainty-aware features  
- Handles multi-label scenarios

### 3. Phase 3: Decoupled Late Fusion
- Uses **Qwen3-8B** as Semantic Scorer  
- Decouples LLM reasoning from ranking core  
- Avoids:
	- Over-reasoning hallucination  
	- Early-fusion data shift  

---

🏆 **Performance:**  
ViMELD achieves State-of-the-Art **F2-Macro = 0.8996** on ALQAC 2025 private test set.

---

## 📂 Repository Structure

```text
.
├── config.py
├── scripts/
│   ├── run_lambda_ablation.py
│   ├── train_meta_learner.py
│   ├── extract_case_study.py
│   └── ...
└── src/
		├── data_prep/
		├── retrieval/
		└── models/
```

---

## ⚙️ Setup & Installation

### 1. Clone repository

```bash
git clone https://github.com/lee-vtruong/legal-retrieval-project.git
cd legal-retrieval-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Note:
> Ensure you have:

* `dgl` (for Graph Neural Networks)
* `vllm` (if running Qwen3 locally)

---

## 💾 Data and Model Weights

To keep this repository lightweight, large files are not included.

### 📊 Data

Download ALQAC 2025 dataset and place at:

```bash
data/raw/ALQAC_2025/
```

### 🧠 Model Checkpoints

Download from:

```
[Your Google Drive / HuggingFace link]
```

Place at:

```bash
experiments/models/
```

---

## 📜 Citation

```bibtex
@inproceedings{vimeld2026,
	title={ViMELD: Vietnamese Multi-view Enhanced Legal Decoupling Framework for Legal Document Retrieval},
	author={Le, Van-Truong and others},
	booktitle={Proceedings of ALQAC},
	year={2026}
}
```