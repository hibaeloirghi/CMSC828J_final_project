import torch
import os
from transformers import pipeline
import datetime
import argparse
import jsonlines
from datasets import load_dataset

def remove_before(text, word):
    parts = text.split(word, 1)
    return parts[1] if len(parts) > 1 else text

if __name__ == "__main__":
    start_time = datetime.datetime.now()

    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_hf", type=str, help="Name of the model on Hugging Face")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cache_dir", type=str, default="/path/to/cache")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train, validation, test)")
    parser.add_argument("--target_language", type=str, default="Moroccan Darija", help="Target translation language")
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("facebook/xnli", "all_languages", cache_dir=args.cache_dir)
    data = dataset[args.split]

    # Initialize translation pipeline
    pipe = pipeline("text-generation", model=args.model_name_hf, torch_dtype=torch.bfloat16, device_map="auto")

    # Open output file
    with jsonlines.open(args.output_path, mode="w") as outfile:
        for line in data:
            # Translate premise
            src = line["premise"]
            prompt = f"Translate English sentence to Moroccan Darija.\n\nEnglish: {src}\nDarija:"

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
            generation = outputs[0]["generated_text"]

            if "<|im_start|>assistant\n " in generation:
                keyword = "<|im_start|>assistant\n "
                generation = remove_before(generation, keyword)

            line["premise_darija"] = generation

            # Translate hypothesis
            src = line["hypothesis"]
            prompt = f"Translate the following sentence from English to Moroccan Darija.\n\nEnglish: {src}\nDarija:"
            messages[1]["content"] = prompt
            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
            generation = outputs[0]["generated_text"]

            if "<|im_start|>assistant\n " in generation:
                generation = remove_before(generation, keyword)

            line["hypothesis_darija"] = generation

            outfile.write(line)

    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")