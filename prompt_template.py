#dyh2111 - prompt template for use with MoeMe
 
 
def prompt_template(input, option, benchmark='pubmedqa'):
    """
    Generate a prompt template based on the specified benchmark, including task descriptions.
    
    Args:
        input (str): The question or primary input text
        option (str): Context or options to include in the prompt
        benchmark (str): The benchmark type to determine prompt format
    
    Returns:
        str: Formatted prompt according to the benchmark's requirements
    """
    # Dictionary of task descriptions and prompt formats
    benchmark_info = {
        'pubmedqa': {
            'description': "Answer biomedical questions using the provided abstract with answers limited to yes, no, or maybe.",
            'template': f"INPUT: {input} CONTEXT: {option} OUTPUT:"
        },
        'medqa': {
            'description': "Simulate taking the US Medical Licensing Examination. Answer a multiple-choice question based on medical knowledge and current practices.",
            'template': f"Question: {input} Options: {option} Answer:"
        },
        'medmcqa': {
            'description': "Answer real-world medical entrance exam questions by selecting the correct answer from multiple choices based on clinical and basic medical science.",
            'template': f"Question: {input} Options: {option} Answer:"
        },
        'emrqa': {
            'description': "Extract the relevant text segment from a given medical context that directly answers an open-ended question.",
            'template': f"Context: {option} Question: {input} Answer:"
        },
        '2012_i2b2': {
            'description': "Identify clinically relevant entities in a sentence from clinical narrative notes and mark them with HTML tags.",
            'template': f"Input Text: {input} Output Text:"
        },
        '2013_ddi': {
            'description': "Predict relationships between two drug entities within a sentence. Identify types of drug-drug interactions.",
            'template': f"INPUT: {input} OUTPUT:"
        },
        'hoc': {
            'description': "Decide which of the Hallmarks of Cancer topics an article's abstract relates to. Articles may relate to multiple topics.",
            'template': f"INPUT: {input} OUTPUT:"
        },
        'mtsample': {
            'description': "Determine the medical specialty or domain of a medical transcription from a list of 40 options.",
            'template': f"INPUT: {input} OUTPUT:"
        },
        'pubmedsum': {
            'description': "Summarize a biomedical literature piece in six sentences.",
            'template': f"INPUT: {option} OUTPUT:"
        },
        'mimic-cxr': {
            'description': "Derive the impression from findings in a radiology report.",
            'template': f"INPUT: {input} OUTPUT:"
        },
        'bionli': {
            'description': "Classify the relationship between a given premise and hypothesis as entailment, or contradiction.",
            'template': f"INPUT: {input} OUTPUT:"
        },
        'mednli': {
            'description': "Classify the relationship between a given premise and hypothesis as entailment, contradiction, or neutral.",
            'template': f"INPUT: {input} OUTPUT:"
        }
    }
    
    # Convert benchmark to lowercase and handle any special cases
    benchmark = benchmark.lower().replace('-', '_')
    
    # Get the benchmark info or default to pubmedqa if not found
    info = benchmark_info.get(benchmark, benchmark_info['pubmedqa'])
    
    # Return the prompt with description and template
    return f"{info['description']}\n{info['template']}"