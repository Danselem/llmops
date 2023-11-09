import os
from comet_ml import API


api = API(api_key=os.environ["COMET_API_KEY"])
COMET_WORKSPACE = os.environ["COMET_WORKSPACE"]


# model name
model_name = "emotion-flan-t5-base"

#get the Model object
model = api.get_model(workspace=COMET_WORKSPACE, model_name=model_name)

# Download a Registry Model:
model.download("1.0.0", "./deploy", expand=True)