import json

# input and output file paths
input_file = "translate_eng_to_dar_llama_3376401.jsonl"
output_file = "results_eng_to_dar.txt"

# Open files
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line)
        
        en_hypothesis = data["hypothesis"]["translation"][4]
        en_premise = data["premise"]["en"]
        dar_hypothesis = data.get("hypothesis_darija", "N/A")
        dar_premise = data.get("premise_darija", "N/A")
        
        result = (
            f"English Hypothesis: {en_hypothesis}\n"
            f"English Premise: {en_premise}\n"
            f"Darija Hypothesis: {dar_hypothesis}\n"
            f"Darija Premise: {dar_premise}\n"
            "-----------------------------------------\n"
        )
        
        # write to the output file
        outfile.write(result)

print(f"Results have been saved to {output_file}")
