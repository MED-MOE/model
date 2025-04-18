# Libraries
import os
import json
import torch
import time
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def prompt_model(model, tokenizer, prompt, max_new_tokens=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def pubmed_prompt(model, tokenizer, pmid, data_file, verbose=False):
    prompt = "Given the QUESTION, return an ANSWER using only [yes,no,maybe]."
    
    # Handle reasoning_required cases differently
    if "reasoning_required_pred" in data_file[pmid] and data_file[pmid]["reasoning_required_pred"] == "yes":
        # Join the contexts list into a single string if it's a list
        if isinstance(data_file[pmid]["CONTEXTS"], list):
            contexts = " ".join(data_file[pmid]["CONTEXTS"])
        else:
            contexts = data_file[pmid]["CONTEXTS"]
            
        # Include context for reasoning-required questions
        question = "QUESTION: " + contexts + " " + data_file[pmid]["QUESTION"]
    else:
        # Use only the question for non-reasoning questions
        question = "QUESTION: " + data_file[pmid]["QUESTION"]
    
    # Combine prompt and question
    full_prompt = prompt + "\n\n" + question + "\n\nANSWER:"
    
    if verbose:
        print(full_prompt)
    
    # Get model output
    full_response = prompt_model(model, tokenizer, full_prompt)

    if verbose:
        print(f"Full Response: {full_response}")
    
    # Extract only the generated part
    answer_part = full_response[len(full_prompt):].strip().lower()
    
    # Parse to get yes, no, or maybe
    if "yes" in answer_part[:20]:
        prediction = "yes"
    elif "no" in answer_part[:20]:
        prediction = "no"
    elif "maybe" in answer_part[:20]:
        prediction = "maybe"
    else:
        # Count occurrences in case the answer isn't at the start
        yes_count = answer_part.count("yes")
        no_count = answer_part.count("no")
        maybe_count = answer_part.count("maybe")
        
        if yes_count > no_count and yes_count > maybe_count:
            prediction = "yes"
        elif no_count > yes_count and no_count > maybe_count:
            prediction = "no"
        else:
            prediction = "maybe"
    
    if verbose:
        print(f"Final Answer: {prediction}")
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
    
    # Load Model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully!")
    
    # Load the dataset
    print(f"Loading PubMedQA data from {args.pubmedqa_path}")
    with open(args.pubmedqa_path, 'r') as f:
        pubmedqa_data = json.load(f)
    
    # Load the ground truth
    print(f"Loading PubMedQA GT from {args.gt_path}")
    with open(args.gt_path, 'r') as f:
        gt_data = json.load(f)
    
    # Get the list of PMIDs to process
    pmids = list(gt_data.keys())
    if args.limit:
        pmids = pmids[:args.limit]
    
    print(f"Starting predictions. Processing {len(pmids)} questions.")
    
    # Initialize predictions dictionary
    preds = {}
    
    # Record start time
    start_time = time.time()
    
    # Process each PMID
    for i, pmid in enumerate(tqdm(pmids, desc="Processing questions")):
        if args.verbose:
            print(f"Question: {i+1}")
        question_start = time.time()
        prediction = pubmed_prompt(model, tokenizer, pmid, pubmedqa_data, verbose=args.verbose)
        preds[pmid] = prediction
        question_elapsed_time = (time.time() - question_start)  # sec
        if args.verbose:
            print(f"Inference Time: {question_elapsed_time:.2f}sec")
    
    # Calculate elapsed time
    elapsed_time = (time.time() - start_time) / 60  # minutes
    
    print("Predictions complete")
    print(f"Elapsed time: {elapsed_time:.2f} mins")
    
    # Save predictions to file
    with open(args.output_file, 'w') as f:
        json.dump(preds, f, indent=2)
    
    print(f"Predictions saved to {args.output_file}")
    

if __name__ == "__main__":
    main()