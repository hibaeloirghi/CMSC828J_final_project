import torch
import argparse
import datetime
import jsonlines
import random
import os
from huggingface_hub.hf_api import HfFolder
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

random.seed(24)

own_cache_dir = "/fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/.cache"
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir

def predict_entailment(pipe, premise, hypothesis):
    prompt = (
        "You are a helpful assistant. Given the following premise and hypothesis, "
        "determine if the premise entails the hypothesis, contradicts it, or neither (neutral). "
        "Respond with only one word: 'entailment', 'contradiction', or 'neutral'.\n\n"
        f"Premise: {premise}\nHypothesis: {hypothesis}\nRespond with only one word for the label:"
    )
    
    generated_text = pipe(prompt, return_full_text=False)[0]["generated_text"]
    
    if 'entailment' in generated_text.lower():
        predicted_label = '0'
    elif 'contradiction' in generated_text.lower():
        predicted_label = '2'
    elif 'neutral' in generated_text.lower():
        predicted_label = '1'
    else:
        predicted_label = 'null'
    
    return generated_text, predicted_label

def main():
    start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_hf", type=str, required=True, help="Name of the model on Hugging Face")
    parser.add_argument("--language", type=str, required=True, help="Language code for the dataset (e.g., 'en', 'ar', 'bg')")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="/path/to/cache")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (train, validation, test)")
    args = parser.parse_args()

    hf_token = "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"
    HfFolder.save_token(hf_token)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_hf, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_hf, cache_dir=args.cache_dir, torch_dtype=dtype)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        torch_dtype=dtype,
        max_new_tokens=10,
        model_kwargs={"temperature": 0.0, "do_sample": False}
    )

    dataset = load_dataset("facebook/xnli", args.language, split=args.split)

    true_labels = []
    predicted_labels = []

    with jsonlines.open(args.output_path, mode="w") as outfile:
        for i, line in enumerate(tqdm(dataset)):
            if i >= 10:  # Process only the first 10 sentences
                break
            premise = line["premise"]
            hypothesis = line["hypothesis"]
            true_label = line["label"]

            raw_output, predicted_label = predict_entailment(pipe, premise, hypothesis)

            print(f"Raw Model Output: {raw_output}")
            print(f"Predicted Label: {predicted_label}\n")

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            line["predicted_label"] = predicted_label
            line["raw_model_output"] = raw_output
            
            outfile.write(line)

    print("\nComparison Table:")
    print("True Label | Predicted Label | Count")
    print("-----------|-----------------|---------")
    for true_label in range(3):
        for pred_label in range(3):
            count = sum((t == true_label) and (p == pred_label) for t, p in zip(true_labels, predicted_labels))
            print(f"{true_label:10d} | {pred_label:15d} | {count:7d}")

    print("\nOverall Accuracy:", sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels))

    end_time = datetime.datetime.now()
    print(f"\nTime elapsed: {end_time - start_time}")

if __name__ == "__main__":
    main()