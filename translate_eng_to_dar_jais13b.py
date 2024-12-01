import torch
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

    # Load dataset with streaming to handle large files efficiently
    dataset = load_dataset("facebook/xnli", "all_languages", split=args.split, streaming=True, cache_dir=args.cache_dir)

    # Load tokenizer and model with trust_remote_code=True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_hf, cache_dir=args.cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        # Set a dedicated pad token if not already present
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Added '[PAD]' as a special pad token.")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_hf,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True  # Trust remote code for loading the model
    )
    model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings for new token

    # Open output file for writing
    with jsonlines.open(args.output_path, mode="w") as outfile:
        count = 0  # Counter to process only 5 pairs

        for line in dataset:
            # Extract the English premise and hypothesis
            premise_en = line["premise"].get("en")
            hypothesis_en = line["hypothesis"]["translation"][line["hypothesis"]["language"].index("en")]

            if not premise_en or not hypothesis_en:
                print("Skipping line due to missing English premise or hypothesis.")
                continue

            # Translate premise
            premise_prompt = f"Translate English sentence to the Moroccan Arabic dialect. Only use the Arabic script to write the translation. Only respond with the translated sentence. Do not include anything written in English.\n\nEnglish: {premise_en}\nDarija:"
            inputs = tokenizer(
                premise_prompt, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")
            inputs["attention_mask"] = inputs["attention_mask"].to("cuda")  # Ensure attention mask is passed
            outputs = model.generate(
                inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                max_new_tokens=256, 
                do_sample=False
            )
            premise_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the translated text (remove the prompt from output)
            premise_darija = premise_translation.split("Darija:")[-1].strip()

            # Translate hypothesis
            hypothesis_prompt = f"Translate English sentence to the Moroccan Arabic dialect. Only use the Arabic script to write the translation. Only respond with the translated sentence. Do not include anything written in English.\n\nEnglish: {hypothesis_en}\nDarija:"
            inputs = tokenizer(
                hypothesis_prompt, return_tensors="pt", padding=True, truncation=True
            ).to("cuda")
            inputs["attention_mask"] = inputs["attention_mask"].to("cuda")  # Ensure attention mask is passed
            outputs = model.generate(
                inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                max_new_tokens=256, 
                do_sample=False
            )
            hypothesis_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the translated text (remove the prompt from output)
            hypothesis_darija = hypothesis_translation.split("Darija:")[-1].strip()

            # Add translations to the output
            line["premise_darija"] = premise_darija
            line["hypothesis_darija"] = hypothesis_darija

            # Write the translated line to the output
            outfile.write(line)

            count += 1
            if count >= 5:  # Stop after 5 pairs
                break

    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")