# set existing experiment
import os
from comet_ml import API, ExistingExperiment

api = API(api_key=os.environ["COMET_API_KEY"])

experiment = ExistingExperiment(api_key=api, 
                previous_experiment="06e487dcae9b4232b3530303145953aa") # get experiment id from comet project.
experiment.log_model("Emotion-T5-Base", "results/checkpoint-7")
experiment.register_model("Emotion-T5-Base")