# CoT_DDI

Chain-of-thought distillation for drug-drug interaction classification.

## Overview

CoT_DDI is a research project focused on distilling pharmacological reasoning from a large language model (Llama-3.3-70B) into a more efficient student model (Qwen2.5-7B) for drug-drug interaction (DDI) prediction tasks. By leveraging chain-of-thought (CoT) prompting and an ensemble-based filtering approach, we achieve strong performance while maintaining computational efficiency.

## Project structure

CoT_DDI/ 
├── README.md 
├── LICENSE 
├── .gitignore 
├── setup_env.sh 
├── activate_env.sh
├── configs/ │ └── [configuration files] 
├── data/ │ └── [datasets] 
├── experiments/ │ └── [experiment results and logs]
├── scripts/ │ └── [utility and automation scripts] 
└── src/ └── [source code]


## Purpose

This project demonstrates how to transfer complex pharmacological reasoning and decision-making capabilities from a large teacher model to a smaller, more practical student model through:
- **Teacher Generation**: Generating detailed reasoning chains from Llama-3.3-70B
- **Judge Ensemble Filtering**: Ensuring quality of generated reasoning through consensus-based filtering
- **Two-Phase Student Fine-Tuning**: Progressively training Qwen2.5-7B for improved DDI prediction

## Key Components

1. **Teacher Generation**
   - Uses Llama-3.3-70B to generate chain-of-thought reasoning for drug-drug interactions
   - Produces detailed explanations of interaction mechanisms and predictions

2. **Judge Ensemble Filtering**
   - Employs multiple judge models to validate generated reasoning
   - Filters high-quality examples for student training
   - Ensures consistency and accuracy of training data

3. **Two-Phase Student Fine-Tuning**
   - Phase 1: Initial fine-tuning on filtered CoT examples
   - Phase 2: Progressive refinement and optimization
   - Results in a compact 7B parameter model with strong reasoning capabilities

## Results

| Configuration | Macro F1 |
|---|---|
| Label-only | 0.9196 |
| Sequential CoT | 0.8475 |

## Installation

### Requirements
- Python 3.8+
- PyTorch
- Transformers library
- Additional dependencies (see `requirements.txt` once available)

### Setup

```bash
git clone https://github.com/Mriyazat/CoT_DDI.git
cd CoT_DDI
pip install -r requirements.txt
