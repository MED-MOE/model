# Imports
import os
import json
import time
import torch
import argparse
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def pubmed_prompt(model_pipeline, pmid, data_file, verbose=False):
    # Define response prompt
    prompt = " Answer yes, no, or maybe:"
    
    # Handle reasoning_required cases
    if "reasoning_required_pred" in data_file[pmid] and data_file[pmid]["reasoning_required_pred"] == "yes":
        contexts = data_file[pmid]["CONTEXTS"]
        if isinstance(contexts, list):
            contexts = " ".join(contexts)
        question = contexts + " " + data_file[pmid]["QUESTION"]
    else:
        question = data_file[pmid]["QUESTION"]
    
    full_prompt = question + prompt
    
    if verbose:
        print(f"Prompt: {full_prompt}")
    
    # Generate output from the model
    output = model_pipeline(full_prompt, max_new_tokens=10, do_sample=False)
    full_response = output[0]["generated_text"]
    
    if verbose:
        print(f"Full Response: {full_response}")
    
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
    
    # Load model and tokenizer
    print("Loading model pipeline...")
    model_pipeline = pipeline("text-generation", model=args.model_path, device_map="auto")
    print("Model loaded successfully!")
    
    # Load dataset
    print(f"Loading PubMedQA data from {args.pubmedqa_path}")
    with open(args.pubmedqa_path, 'r') as f:
        pubmedqa_data = json.load(f)
    
    # Load ground truth
    print(f"Loading PubMedQA GT from {args.gt_path}")
    with open(args.gt_path, 'r') as f:
        gt_data = json.load(f)
    
    # Prepare list of PMIDs
    pmids = list(gt_data.keys())
    if args.limit:
        pmids = pmids[:args.limit]
    
    print(f"Starting predictions. Processing {len(pmids)} questions.")
    
    preds = {}
    start_time = time.time()
    
    # Inference loop
    for i, pmid in enumerate(tqdm(pmids, desc="Processing questions")):
        if args.verbose:
            print(f"\nQuestion {i+1}/{len(pmids)} — PMID: {pmid}")
        question_start = time.time()
        
        try:
            prediction = pubmed_prompt(model_pipeline, pmid, pubmedqa_data, verbose=args.verbose)
        except Exception as e:
            print(f"Error on PMID {pmid}: {e}")
            prediction = "maybe"
        
        preds[pmid] = prediction
        
        if args.verbose:
            question_elapsed = time.time() - question_start
            print(f"Inference Time: {question_elapsed:.2f} sec")
    
    elapsed_time = (time.time() - start_time) / 60
    print(f"\nPredictions complete. Elapsed time: {elapsed_time:.2f} minutes.")
    
    # Save predictions
    with open(args.output_file, 'w') as f:
        json.dump(preds, f, indent=2)
    
    print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
