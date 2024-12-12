import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime
import argparse
import jsonlines
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

def predict_entailment(model, tokenizer, premise, hypothesis):
    prompt = f"Given the following premise and hypothesis, determine if the premise entails the hypothesis, contradicts it, or neither (neutral). Respond with only one of these labels: entailment, contradiction, or neutral.\n\nPremise: {premise}\nHypothesis: {hypothesis}\nLabel:"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=10, do_sample=False)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Label:")[-1].strip().lower()
    
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    return label_map.get(prediction, 1)  # Default to neutral if prediction is not recognized

if __name__ == "__main__":
    start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_hf", type=str, default="meta-llama/Llama-2-7b-chat-hf", help="Name of the model on Hugging Face")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cache_dir", type=str, default="/path/to/cache")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (train, validation, test)")
    args = parser.parse_args()

    dataset = load_dataset("facebook/xnli", "en", split=args.split, streaming=True, cache_dir=args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_hf, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_hf,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))

    true_labels = []
    predicted_labels = []

    with jsonlines.open(args.output_path, mode="w") as outfile:
        for line in tqdm(dataset):
            premise = line["premise"]
            hypothesis = line["hypothesis"]
            true_label = line["label"]

            predicted_label = predict_entailment(model, tokenizer, premise, hypothesis)

            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

            line["predicted_label"] = predicted_label
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