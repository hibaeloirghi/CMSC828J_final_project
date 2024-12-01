import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime
import argparse
import jsonlines
from datasets import load_dataset

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
    dataset = load_dataset("facebook/xnli", "all_languages", split=args.split, streaming=True, cache_dir=args.cache_dir)

    # Load tokenizer and add a padding token if not defined
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_hf, cache_dir=args.cache_dir)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use the end-of-sequence token as the padding token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_hf,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Open output file for writing
    with jsonlines.open(args.output_path, mode="w") as outfile:
        max_pairs = 5  # Limit to the first 5 pairs
        processed_pairs = 0

        for line in dataset:
            if processed_pairs >= max_pairs:
                break  # Stop processing after 5 pairs

            # Extract English premise and hypothesis
            premise_en = line["premise"]["en"]
            hypothesis_en = line["hypothesis"]["en"]

            # Prepare prompts for translation
            premise_prompt = f"Translate the following sentence from English to Moroccan Darija:\n\nEnglish: {premise_en}\nDarija:"
            hypothesis_prompt = f"Translate the following sentence from English to Moroccan Darija:\n\nEnglish: {hypothesis_en}\nDarija:"

            # Tokenize and generate translations
            inputs = tokenizer([premise_prompt, hypothesis_prompt], return_tensors="pt", padding=True, truncation=True).to("cuda")
            outputs = model.generate(inputs["input_ids"], max_new_tokens=256, do_sample=False)
            translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Add translated results back to the original data
            translated_line = {
                "premise_en": premise_en,
                "premise_md": translations[0].strip(),
                "hypothesis_en": hypothesis_en,
                "hypothesis_md": translations[1].strip(),
                "label": line["label"]
            }
            outfile.write(translated_line)

            processed_pairs += 1

    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")