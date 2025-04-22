'''
This file has been updated according to the prompt template here:
https://physionet.org/content/me-llama/1.0.0/
INPUT: {Text} CONTEXT: {Text} OUTPUT:
'''
# Imports
import os
import json
import time
import torch
import argparse
import datetime
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Function formats and prepares text for input to model
def pubmed_prompt(model_pipeline, pmid, data_file, verbose=False):
    
    # Define response prompt
    # INPUT: {Text} CONTEXT: {Text} OUTPUT:
    prompt = "TASK: Your task is to answer biomedical questions using the given abstract. Only output yes, no, or maybe as answer. "
    
    # Handle reasoning_required cases
    contexts = ""
    
    if "reasoning_required_pred" in data_file[pmid] and data_file[pmid]["reasoning_required_pred"] == "yes":
        contexts = data_file[pmid]["CONTEXTS"]
        
        if isinstance(contexts, list):
            contexts = " ".join(contexts)
        question = contexts + " " + data_file[pmid]["QUESTION"]
    question = data_file[pmid]["QUESTION"]
    
    full_prompt = f"{prompt} INPUT: {contexts} {question} OUTPUT:"
    
    if verbose:
        print(f"Prompt: {full_prompt}")
    
    # Generate output from the model
    output = model_pipeline(full_prompt, num_return_sequences=1)
    full_response = output[0]['generated_text']
    
    #if verbose:
     #   print(f"Full Response: {full_response}")
    
    # Extract generated answer
    answer_part = full_response[len(full_prompt):].strip().lower()
    
    # Heuristic for answer parsing
    if "yes" in answer_part[:20]:
        prediction = "yes"
    elif "no" in answer_part[:20]:
        prediction = "no"
    elif "maybe" in answer_part[:20]:
        prediction = "maybe"
    else:
        yes_count = answer_part.count("yes")
        no_count = answer_part.count("no")
        maybe_count = answer_part.count("maybe")
        if yes_count > no_count and yes_count > maybe_count:
            prediction = "yes"
        elif no_count > yes_count and no_count > maybe_count:
            prediction = "no"
        else:
            prediction = "maybe"
    true_answer = data_file[pmid]["final_decision"]
    if verbose:
        print(f"True Answer: {true_answer} | Model Answer: {prediction}")
    
    return prediction

def main():
    parser = argparse.ArgumentParser(description='Run PubMedQA benchmark with MeLLaMA model')
    parser.add_argument('--model_path', type=str, default="/mnt/models/MeLLaMA-13B", 
                        help='Path to the model')
    parser.add_argument('--pubmedqa_path', type=str, 
                        default="/home/dyh2111/moeme/model/pubmedqa/data/ori_pqal.json",
                        help='Path to PubMedQA data file')
    parser.add_argument('--gt_path', type=str, 
                        default="/home/dyh2111/moeme/model/pubmedqa/data/test_ground_truth.json",
                        help='Path to ground truth data file')
    parser.add_argument('--output_file', type=str, default="mellama_pubmedqa_predictions.json",
                        help='Output file for predictions')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of questions to process (for testing)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()
    
    # Load PubMedQA dataset
    with open(args.pubmedqa_path, 'r') as f:
        pubmedqa_data = json.load(f)
    
    # If output file exists, rename it with timestamp
    if os.path.exists(args.output_file):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.splitext(args.output_file)[0]}_{timestamp}{os.path.splitext(args.output_file)[1]}"
        os.rename(args.output_file, backup_filename)
        print(f"Renamed existing file to {backup_filename}")
    
    # Start with fresh results
    results = {}
    
    # Initialize model
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50)
    
    # Get PMIDs to process
    pmids = list(pubmedqa_data.keys())
    if args.limit:
        pmids = pmids[:args.limit]
    
    # Set up progress bar
    pbar = tqdm(pmids, desc="Processing PubMedQA entries")
    
    # Process questions
    correct = 0
    total = 0
    
    for pmid in pbar:
        # Process question
        prediction = pubmed_prompt(pipe, pmid, pubmedqa_data, verbose=args.verbose)
        
        # Update results
        results[pmid] = prediction
        
        # Write results to file after each prediction
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Update accuracy metrics
        true_answer = pubmedqa_data[pmid]["final_decision"]
        if prediction == true_answer:
            correct += 1
        total += 1
        
        accuracy = correct / total if total > 0 else 0
        pbar.set_postfix({"Accuracy": f"{accuracy:.4f}"})
    
        # Final accuracy calculation
        print(f"\nCurrent accuracy: {correct}/{total} = {correct/total:.4f}")

if __name__ == "__main__":
    main()