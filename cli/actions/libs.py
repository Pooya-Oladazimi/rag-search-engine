from actions.vars import DATASET
import json


def load_dataset():
    with open(DATASET, "r") as f:
        dataset = json.loads(f.read())
        return dataset["movies"]
