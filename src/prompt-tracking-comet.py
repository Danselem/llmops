# !/bin/python
# Author: Daniel Egbo
# Description: This script does tracking of LLM Prompts with Comet LLM.
# Date: 31 October 2023

# Import necessary libraries if needed
import openai
import os
import IPython
import json
import pandas as pd
import numpy as np
import comet_llm
import urllib
import time

# API configuration
openai.api_key = os.environ["OPENAI_API_KEY"]

# set up comet
COMET_API_KEY = os.environ["COMET_API_KEY"]
COMET_WORKSPACE = os.environ["COMET_WORKSPACE"]

def get_completion(messages, model="gpt-3.5-turbo", 
                   temperature=0, max_tokens=300):
    """
    The function helps to generate the final results from the model after calling the OpenAI API.
    Args:
        messages (str): A text in string format.
        model: The LLM model e.g.  gpt-3.5-turbo, llama2-7b.
        temperature: Model temperature
        max_tokens: Number of context tokens

    Returns:
        str: The out of the prompt results.
    
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]


# print markdown
def print_markdown(text):
    """Prints text as markdown"""
    IPython.display.display(IPython.display.Markdown(text))
    
def get_predictions(prompt_template, inputs):
    """
    This is a helper function to obtain the final predictions from 
    the model given a prompt template (e.g., zero-shot or few-shot) and 
    the provided input data.
    
    """

    responses = []

    for i in range(len(inputs)):
        messages = messages = [
            {
                "role": "system",
                "content": prompt_template.format(input=inputs[i])
            }
        ]
        response = get_completion(messages)
        responses.append(response)

    return responses
    

### Few-Shot

# function to define the few-shot template
def get_few_shot_template(few_shot_prefix, few_shot_suffix, few_shot_examples):
    return few_shot_prefix + "\n\n" + "\n".join(
        [ "Abstract: "+ex["abstract"] + "\n" + "Tags: " + str(
            ex["tags"]) + "\n" for ex in few_shot_examples]) + "\n\n" + few_shot_suffix

# function to sample few shot data
def random_sample_data (data, n):
    return np.random.choice(few_shot_data, n, replace=False)

### Zero-Shot

zero_shot_template = """
Your task is extract model names from machine learning paper abstracts. 
Your response is an an array of the model names in the format [\"model_name\"]. 
If you don't find model names in the abstract or you are not sure, return [\"NA\"]

Abstract: {input}
Tags:
"""
    
# load validation data from GitHub
f = urllib.request.urlopen("https://raw.githubusercontent.com/comet-ml/comet-llmops/main/data/article-tags.json")
val_data = json.load(f)

# load few shot data from GitHub
f = urllib.request.urlopen("https://raw.githubusercontent.com/comet-ml/comet-llmops/main/data/few_shot.json")
few_shot_data = json.load(f)


# the few-shot prefix and suffix
few_shot_prefix = """Your task is to extract model names from machine learning paper abstracts. 
Your response is an an array of the model names in the format [\"model_name\"]. 
If you don't find model names in the abstract or you are not sure, return [\"NA\"]"""

few_shot_suffix = """Abstract: {input}\nTags:"""

# load 3 samples from few shot data
few_shot_template = get_few_shot_template(few_shot_prefix, few_shot_suffix, random_sample_data(few_shot_data, 3))



### Get Predictions
# get the predictions

abstracts = [val_data[i]["abstract"] for i in range(len(val_data[3:6]))]
few_shot_predictions = get_predictions(few_shot_template, abstracts)

# pausing the OpenAI api call for a minute
time.sleep(60)
zero_shot_predictions = get_predictions(zero_shot_template, abstracts)
expected_tags = [str(val_data[i]["tags"]) for i in range(len(val_data[3:6]))]


# log the predictions in Comet along with the ground truth for comparison

# initialize comet
comet_llm.init(COMET_API_KEY, COMET_WORKSPACE, project="ml-paper-tagger-prompts2")

# log the predictions
for i in range(len(expected_tags)):
    # log the few-shot predictions
    comet_llm.log_prompt(
        prompt=few_shot_template.format(input=abstracts[i]),
        prompt_template=few_shot_template,
        output=few_shot_predictions[i],
        tags = ["gpt-3.5-turbo", "few-shot"],
        metadata = {
            "expected_tags": expected_tags[i],
            "abstract": abstracts[i],
        }
    )

    # log the zero-shot predictions
    comet_llm.log_prompt(
        prompt=zero_shot_template.format(input=abstracts[i]),
        prompt_template=zero_shot_template,
        output=zero_shot_predictions[i],
        tags = ["gpt-3.5-turbo", "zero-shot"],
        metadata = {
            "expected_tags": expected_tags[i],
            "abstract": abstracts[i],
        }
    )

print("Few shot predictions")
print(few_shot_predictions)
print("\n\nZero shot predictions")
print(zero_shot_predictions)
print("\n\nExpected tags")
print(expected_tags)