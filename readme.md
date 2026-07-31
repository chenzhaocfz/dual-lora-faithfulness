# Sequential Dual-LoRA Reasoning Faithfulness Thesis

**Author:** Chen Zhao  
**Institution:** Sheridan College
**Base Model:** Qwen-3-8B (4-bit QLoRA)  
**Hardware:** NVIDIA RTX 4070 (12GB VRAM)

---

## Overview
This repository contains the codebase for investigating whether physically separating "Thinking" and "Answering" into runtime-swapped LoRA adapters improves Chain-of-Thought reasoning faithfulness compared to standard monolithic models.

---

## Current Structure
```text
dual-lora-faithfulness/
├── .env                        
├── .gitignore                 
├── README.md                   
├── requirements.txt            
├── data/
│   ├── raw/                    # Raw GSM8K / GSM-Symbolic downloads
│   └── processed/              # Filtered JSON fine-tuning sets
└── src/
    ├── test-qwen3.py           # VRAM & Unsloth GPU test script
    ├── oracle-generation.py    # DeepSeek V4 API synthetic generator
    └── load-datasets.py        # load and filter GSM8K / GSM-Symbolic 