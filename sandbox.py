from transformers import AutoTokenizer, OPTForCausalLM
import torch
import os
import sys
import pickle as pk
import pandas as pd
import numpy as np
import psutil


default_dtype = torch.bfloat16
torch.set_default_dtype(default_dtype)
default_device = 'gpu'
max_memory["cpu"] = psutil.virtual_memory().available


tokenizer = AutoTokenizer.from_pretrained("RWTH-TIME/galactica-125m-f-term-classification")


model = OPTForCausalLM.from_pretrained("RWTH-TIME/galactica-125m-f-term-classification", torch_dtype=default_dtype, low_cpu_mem_usage=True,
                                           device_map="auto", max_memory=max_memory)






def generate(prompt, model, tokenizer, max_pred_tokens=10, decode=True, enforce_no_repetition=True, ignore_eos_token=True):
    """ generates FTERM classifications for a given patent abstract  
 
    prompts a given model returns comma seperated FTERMS
 
    Parameters
    ------------
        prompt: string
            A patent abstract text or technology description
        model: transformers.models.opt.modeling_opt.OPTForCausalLM
            The transformer model used for classification
        tokenizer: tokenizer
            The according tokenizer for the model
        max_pred_tokens: int, Optional
            The maximum number of patent classes to be predicted
        decode: boolean, Optional
            The output is either a decoded list of text classes or the not-decoded token numbers
        enforce_no_repetition: boolean, Optional
            If True, inhibits the repeated prediction of the same FTERM class and instead predicts the next most probable class
        ignore_eos_token: boolean, Optional
            If True, enforces the prediction of max_pred_tokens and ignores the models output of eos_token 
    Return
    -----------
        predictions : string
            A list of FTERM classes for the given prompt
    """
    # adding the Start F-Term Token to the prompt to beginng the prediction of F-Terms
    prompt += '<START F-TERMS>'

    # Converting the prompt to tokens
    eos_token_id = tokenizer.eos_token_id
    if ignore_eos_token==True:
        eos_token_id=-999
    tokenized = tokenizer(prompt, return_tensors='pt')
    prompt_tokens = tokenized['input_ids'][:,:-1]
    attention_mask = tokenized['attention_mask'][:, :-1]

    # Generating the F-Terms
    current_token = -100
    predictions = []
    while current_token != eos_token_id and len(predictions) < max_pred_tokens:

        # Model Call
        output = model(prompt_tokens, attention_mask, output_attentions=False, output_hidden_states=False, return_dict=True)
        logits = output['logits']
        # torch.sort function returns the values and indices of the latest prediction.
        current_token_predictions=torch.sort(logits[0][-1], dim=-1, descending=True)
        i=0
        # the most likely item (first item of the list) is picked
        current_token=current_token_predictions[1][i]+50000
        # if enforce_no_repitition is set True, a while loop checks for dublicates with existing predictions and picks the next most likely prediction
        if enforce_no_repetition==True or (current_token==tokenizer.eos_token_id and ignore_eos_token==True):
            while current_token in predictions or (current_token==tokenizer.eos_token_id and ignore_eos_token==True):
                #current_token already predicted, picking next one"
                i+=1
                current_token=current_token_predictions[1][i]+50000
            
        predictions.append(current_token)
        # Adding the prediction to the input sequence to predict the new token.
        prompt_tokens = torch.cat([prompt_tokens, torch.tensor([[current_token]])], -1)
        # Attention mask has to be updates as well
        attention_mask = torch.cat([attention_mask, attention_mask[:,-1:]], -1)
    if decode:
        predictions = tokenizer.decode(predictions)
        return predictions
    else: 
        return predictions
    
abstract="PROBLEM TO BE SOLVED: To enable biological rhythm of small animals to be regulated. <P>SOLUTION: The cushion 1 for small animal includes a base part 2 on which the small animal can lay the body, a swelled part 3 formed on the base part 2, a light-radiating part 5 for irradiating light to the small animal lying on the base part 2, and a control part 8 for switching the light irradiated from the light-radiating part 5 according to the time regulated based on a previously set light pattern. For example, the light-irradiating part 5 includes a light source part 6 that includes a light source in the inside, and a light-inlet part 7 for introducing the light from the light source, while emitting light."

generate(abstract, model, tokenizer)