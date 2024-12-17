import json
import re

def clean_text(text):
    # remove Latin characters
    text = re.sub(r'[a-zA-Z]', '', text)
    # remove line breaks
    text = text.replace('\n', ' ')
    # collapse multiple spaces into a single space
    text = re.sub(r'\s+', ' ', text)
    # remove redundant punctuation
    text = re.sub(r'([.,:;!?-])\1+', r'\1', text)
    text = re.sub(r'([.,:;!?-])[\s\.,:;!?-]*', r'\1', text)
    return text.strip()

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):  
            #if i >= 5:  # ok star out with 5 to test
            #    break
            data = json.loads(line.strip())
            
            # clean up premise and hypothesis
            premise = clean_text(data.get("premise_darija", ""))
            hypothesis = clean_text(data.get("hypothesis_darija", ""))
            
            processed_data = {
                "premise": premise,
                "hypothesis": hypothesis,
                "label": data.get("label", None)
            }
            
            outfile.write(json.dumps(processed_data, ensure_ascii=False) + '\n')


# paths
input_file = "/fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/output/translate_ar_to_dar_llama2_7b_3483915.jsonl"
output_file = "/fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/output/clean_translate_ar_to_dar_llama2_7b_3483915.jsonl"

process_jsonl(input_file, output_file)