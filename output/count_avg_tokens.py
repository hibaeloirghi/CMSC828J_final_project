import json

def count_tokens(text):
    """
    Count the number of tokens in a given string.
    Tokens are defined as the space-separated words in the string.
    Args:
        text (str): The input text.
    Returns:
        int: The number of tokens.
    """
    return len(text.split())

def calculate_avg_tokens(file):
    premise_token_counts = []
    hypothesis_token_counts = []

    with open(file, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line.strip())
            premise = data.get("premise", "")
            hypothesis = data.get("hypothesis", "")
            
            premise_token_counts.append(count_tokens(premise))
            hypothesis_token_counts.append(count_tokens(hypothesis))
    
    # average tokens for the current file
    avg_premise_tokens = sum(premise_token_counts) / len(premise_token_counts) if premise_token_counts else 0
    avg_hypothesis_tokens = sum(hypothesis_token_counts) / len(hypothesis_token_counts) if hypothesis_token_counts else 0
    
    return avg_premise_tokens, avg_hypothesis_tokens

if __name__ == "__main__":
    # paths
    files = [
        "/fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/output/clean_translate_ar_to_dar_llama2_7b_3483915.jsonl",
        "/fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/output/clean_translate_eng_to_dar_llama2_7b_3483866.jsonl",
        "/fs/classhomes/fall2024/cmsc723/c7230002/CMSC828J_final_project/human_translated_30_pairs.jsonl"
    ]
    
    # print average tokens for each file
    for file in files:
        avg_premise, avg_hypothesis = calculate_avg_tokens(file)
        print(f"File: {file}")
        print(f"Average number of tokens in premise: {avg_premise:.1f}")
        print(f"Average number of tokens in hypothesis: {avg_hypothesis:.1f}")
        print()
