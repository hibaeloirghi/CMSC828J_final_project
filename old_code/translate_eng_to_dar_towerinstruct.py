import torch
from transformers import pipeline
import datetime
import argparse
import jsonlines
from datasets import load_dataset

def translate_text(pipe, text, target_language):
    messages = [
        {"role": "user", "content": f"Translate the following English sentence to {target_language}. Respond only with the translated sentence in Arabic script.\nEnglish: {text}\nDarija:"},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=False)
    return outputs[0]["generated_text"].split("Darija:")[-1].strip()

if __name__ == "__main__":
    start_time = datetime.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_hf", type=str, default="Unbabel/TowerInstruct-7B-v0.2", help="Name of the model on Hugging Face")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--cache_dir", type=str, default="/path/to/cache")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train, validation, test)")
    parser.add_argument("--target_language", type=str, default="Moroccan Darija", help="Target translation language")
    args = parser.parse_args()

    dataset = load_dataset("facebook/xnli", "all_languages", split=args.split, streaming=True, cache_dir=args.cache_dir)

    pipe = pipeline("text-generation", model=args.model_name_hf, torch_dtype=torch.bfloat16, device_map="auto")

    with jsonlines.open(args.output_path, mode="w") as outfile:
        for line in dataset:
            premise_en = line["premise"].get("en")
            hypothesis_en = line["hypothesis"]["translation"][line["hypothesis"]["language"].index("en")]

            if not premise_en or not hypothesis_en:
                print("Skipping line due to missing English premise or hypothesis.")
                continue

            premise_darija = translate_text(pipe, premise_en, args.target_language)
            hypothesis_darija = translate_text(pipe, hypothesis_en, args.target_language)

            line["premise_darija"] = premise_darija
            line["hypothesis_darija"] = hypothesis_darija
            outfile.write(line)

    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")
