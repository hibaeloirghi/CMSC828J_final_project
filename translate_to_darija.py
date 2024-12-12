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

own_cache_dir = "/fs/clip-scratch/eloirghi/CMSC828J_final_project/.cache"
os.environ["HF_HOME"] = own_cache_dir
os.environ["HF_DATASETS"] = own_cache_dir

def translate_to_moroccan_arabic(pipe, text):
    prompt = (
        "You are a helpful translator. Translate the following text to Moroccan Arabic. "
        "Provide only the translation without any explanations.\n\n"
        f"Text: {text}\nTranslation:"
    )

    generated_text = pipe(prompt, return_full_text=False)[0]["generated_text"]
    return generated_text.strip()

def main():
    start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_hf", type=str, required=True, help="Name of the model on Hugging Face")
    parser.add_argument("--source_language", type=str, required=True, help="Source language code (e.g., 'en', 'fr', 'es')")
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
        max_new_tokens=100,
        model_kwargs={"temperature": 0.7, "do_sample": True}
    )

    # Load an appropriate dataset for translation
    dataset = load_dataset("opus100", f"{args.source_language}-ar", split=args.split)

    with jsonlines.open(args.output_path, mode="w") as outfile:
        for i, line in enumerate(tqdm(dataset)):
            if i >= 10:  # Process only the first 10 sentences
                break
            source_text = line[args.source_language]
            translation = translate_to_moroccan_arabic(pipe, source_text)
            
            print(f"Source: {source_text}")
            print(f"Translation: {translation}\n")

            outfile.write({
                "source": source_text,
                "translation": translation
            })

    end_time = datetime.datetime.now()
    print(f"\nTime elapsed: {end_time - start_time}")

if __name__ == "__main__":
    main()