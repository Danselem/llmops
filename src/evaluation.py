import os
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import transformers
import pandas as pd
import comet_llm
from comet_ml import API, ExistingExperiment
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# set random seed
transformers.set_seed(35)

COMET_WORKSPACE = "danselem" os.environ['COMET_WORKSPACE']
COMET_API_KEY = os.environ['COMET_API_KEY']


# Download model from registry:

api = API(api_key=COMET_API_KEY)

# model name
model_name = "Emotion-T5-Base"

#get the Model object
model = api.get_model(workspace=COMET_WORKSPACE, model_name=model_name)

# Download a Registry Model:
model.download("1.0.0", "./deploy", expand=True)

# load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./deploy/checkpoint-7")
tokenizer = AutoTokenizer.from_pretrained("./deploy/checkpoint-7/")

emotion_datapath = "https://raw.githubusercontent.com/comet-ml/comet-llmops/main/data/merged_training_sample_prepared_valid.jsonl"

emotion_dataset_val_temp = pd.read_json(path_or_buf=emotion_datapath, lines=True)
emotion_dataset_test = emotion_dataset_val_temp.iloc[int(len(emotion_dataset_val_temp)/2):]


# for comet logging
comet_llm.init(project="emotion-evaluation")

# prompt prefix
prefix = "Classify the provided piece of text into one of the following emotion labels.\n\nEmotion labels: [
    'anger', 'fear', 'joy', 'love', 'sadness', 'surprise']\n\nText:"

# prepare prompts
prompts = [{"prompt": row.prompt.strip("\n\n###\n\n"
                                       ) + "\n\n" + "Emotion output:", "completion": row.completion.strip("\n").strip(
                                           " ")} for index, row in emotion_dataset_test.iterrows()]

# expected results to log
actual_completions = [prompt["completion"] for prompt in prompts]

# the results from the fine-tuned model
finetuned_completions = []

for prompt in prompts:

    # finetuned model outputs
    input_ids = tokenizer.encode(prefix + prompt["prompt"], return_tensors="pt")
    output = model.generate(input_ids, do_sample=True, max_new_tokens=1, temperature=0.1)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True).strip("<pad>").strip(" ")
    finetuned_completions.append(output_text)

    # log the prompts
    comet_llm.log_prompt(
        prompt = prefix + prompt["prompt"],
        tags = ["flan-t5-base", "fine-tuned"],
        metadata = {
            "model_name": "flan-t5-base",
            "temperature": 0.1,
            "expected_output": prompt["completion"],
        },
        output = output_text
    )

    # exercise: log zero-shot and few-shot results with GPT-3.5-Turbo and GPT-4 and compare with your fine-tuned model
    
    
# confusion matrix (logged to experiments as well)

# map completion labels to integers
completion_map = {
    "anger": 0,
    "fear": 1,
    "joy": 2,
    "love": 3,
    "sadness": 4,
    "surprise": 5
}

# mapper back to string labels
completion_map_string = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise"
}

actual_completions_int = [completion_map[completion] for completion in actual_completions]
finetuned_completions_int = [1 if completion == "nightmare" else completion_map[completion] for completion in finetuned_completions]

cm = confusion_matrix(actual_completions_int, finetuned_completions_int)

# plot confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=0.5, square=True, cmap="Blues_r")

# add emotion labels to confusion matrix
plt.ylabel("Actual label")
plt.xlabel("Predicted label")

# annotate the confusion matrix with completion labels
tick_marks = [i for i in range(len(completion_map_string))]
plt.xticks(tick_marks, list(completion_map_string.values()), rotation="vertical")
plt.yticks(tick_marks, list(completion_map_string.values()), rotation="horizontal")



experiment = ExistingExperiment(api_key=COMET_API_KEY, previous_experiment="097ab78e6e154f24b8090a1a7dd6abb8")
experiment.log_confusion_matrix(actual_completions_int, finetuned_completions_int, 
                                labels=list(completion_map_string.values()))
