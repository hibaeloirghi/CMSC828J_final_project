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

    # Load tokenizer and model
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
        for line in dataset:
            # Extract the English premise and hypothesis
            premise_en = line["premise"].get("en")
            hypothesis_en = line["hypothesis"]["translation"][line["hypothesis"]["language"].index("en")]

            if not premise_en or not hypothesis_en:
                print("Skipping line due to missing English premise or hypothesis.")
                continue

            # Translate premise
            # first attempt: it included disclaimers about accuracy of translation            
            #premise_prompt = f"Translate English sentence to Moroccan Darija.\n\nEnglish: {premise_en}\nDarija:"
            # second attempt: it included output with darija written in latin or Amazigh script
            #premise_prompt = f"Translate English sentence to Moroccan Arabic (Darija). Only respond with the translation output.\n\nEnglish: {premise_en}\nDarija:"
            # third attempt: problem (check output 3376179) still yields Note about arabic dialect in English
            #premise_prompt = f"Translate English sentence to the Moroccan Arabic dialect. Only respond with the translation output written in Arabic script.\n\nEnglish: {premise_en}\nDarija:"
            # fourth attempt
            premise_prompt = f"Translate the following English sentence to Moroccan Arabic. Only respond with the translated sentence in Arabic letters.\n\nEnglish: {premise_en}\nMoroccan Arabic:"

            inputs = tokenizer(premise_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
            outputs = model.generate(inputs["input_ids"], max_new_tokens=256, do_sample=True)
            premise_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the translated text (remove the prompt from output)
            premise_darija = premise_translation.split("Darija:")[-1].strip()

            # Translate hypothesis
            hypothesis_prompt = f"Translate the following English sentence to Moroccan Arabic. Only respond with the translated sentence in Arabic letters.\n\nEnglish: {hypothesis_en}\nMoroccan Arabic:"
            inputs = tokenizer(hypothesis_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
            outputs = model.generate(inputs["input_ids"], max_new_tokens=256, do_sample=True)
            hypothesis_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the translated text (remove the prompt from output)
            hypothesis_darija = hypothesis_translation.split("Darija:")[-1].strip()

            # Add translations to the output
            line["premise_darija"] = premise_darija
            line["hypothesis_darija"] = hypothesis_darija

            # Write the translated line to the output
            outfile.write(line)

            #count += 1
            #if count >= 5010:  # Stop after 5 pairs
            #    break

    end_time = datetime.datetime.now()
    print(f"Time elapsed: {end_time - start_time}")