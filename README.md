# CMSC828J_final_project
Final Project for CMSC 828J


This is the official repository for "[Translating the XNLI RTE pairs to the Moroccan Arabic Dialect (Darija) using a LLM].

## Abstract

Hundreds of millions in the Middle East and North Africa (MENA) and worldwide write and speak using one of the many variations of Arabic known as dialects. These dialects lack standardized spelling and are not recognized as official languages in any of the Arabic-speaking countries in the MENA region. To further develop robust and useful Natural Language Processing (NLP) tools and language models for these underserved populations, I turned to Natural Language Inference (NLI) and augmented the XNLI dataset with the Moroccan Arabic Dialect (Darija) using a large language model - Llama2-7B to perform Machine Translation (MT). Further inspection of the MT output reveals linguistic weaknesses and a large gap in the model’s fluency in Darija which is spoken by roughly 40 million people in Morocco. Additionally, I also evaluated the model’s NLI capabilities on XNLI pairs and manually translated 30 XNLI pairs into Darija to provide gold standards for future research. I hope that this limited work will inspire further research into the fluency of several state-of-the-art models in Arabic dialects and other widely spoken languages and dialects around the world.

## Getting Started
**Get access to Llama2-7B model from Huggingface**

If you want to reproduce the experiments in this paper, please sign up for access to the Llama2-7B model on Huggingface: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf. To login in terminal, enter:
```
huggingface-cli login
```
then enter your Huggingface private key beginning with "hf_".

**Get Code**
```
git clone https://github.com/hibaeloirghi/CMSC828J_final_project
```
**Build Environment**
```
cd CMSC828J_final_project
conda create -n CMSC828J_final_project python=3.10
conda activate CMSC828J_final_project
pip install -r requirements.txt
```

## Run things!
I provide bash scripts to run the Machine Translation (MT) and RTE label prediction python scripts. This assumes familiarity with the CLIP lab's Nexus GPU cluster. Please note that you will have to adjust these scripts as needed if you're running computations on a different cluster. 

## Paper Contributions
- [human_translated_30_pairs.jsonl](human_translated_30_pairs.jsonl): 30 XNLI pairs (60 sentences) translated from English to Darija (Moroccan Arabic) by me 😊
- [clean_translate_ar_to_dar_llama2_7b_3483915.jsonl](output/clean_translate_ar_to_dar_llama2_7b_3483915.jsonl): Output of Llama2-7B MT from Modern Standard Arabic (MSA) to Darija
- [clean_translate_eng_to_dar_llama2_7b_3483866.jsonl](output/clean_translate_eng_to_dar_llama2_7b_3483866.jsonl): Output of Llama2-7B MT from English to Darija
- [predict_label_ar_llama2_7b_3389728.jsonl](output/predict_label_ar_llama2_7b_3389728.jsonl): Output of Llama2-7B RTE task on MSA XNLI pairs
- [predict_label_en_llama2_7b_3389727.jsonl](output/predict_label_en_llama2_7b_3389727.jsonl): Output of Llama2-7B RTE task on English XNLI pairs


