import openai
import os
import json
import ast
import time
import comet_llm


from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('../.env')
load_dotenv(dotenv_path=dotenv_path) 

# API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
COMET_API_KEY = os.getenv("COMET_API_KEY")
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")


# completion function
def get_completion(messages, model="gpt-3.5-turbo", temperature=0.3, max_tokens=700):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]

prompt = """
Your task is: to act as a professional sales marketer and draft a sales email \
    marketing content for any random product or service of your choice, \
        ensure the content contains a call to action in the body, return 
        the requested information in the section delimited by ### ###. Format the output as a JSON object.


###
Subject: the subject of the product or service you want to market
Product: Name of the product
Tag: indicate whether the item is a product or service
Body: the body of the email with full contents.
Length: contain the number of words in the body of the text.
###
"""

message = [
    {
        "role": "user",
        "content": prompt
    }
]

# output = ast.literal_eval(get_completion(message))

# Serializing json
# json_object = json.dumps(output, indent=4)
 
# Writing to sample.json
data ="mailmark.json"

with open(data, "w") as outfile:
    json.dump([], outfile)

with open(data, "r") as file:
        existing_data = json.load(file)

for i in range(1000):
    print(i)
    output = eval(get_completion(message, temperature=1.2,max_tokens=1000))
    # output = ast.literal_eval(get_completion(message, temperature=1.2,max_tokens=1000))
    # print(type(output))
    # print(output)
    # Serializing json
    # json_object = json.dumps(output, indent=4)
    # Writing to sample.json
    with open(data, "w") as file:
        
        existing_data.append(output)
        json.dump(existing_data, file, indent=4)
        # outfile.write(json_object)
    time.sleep(20)