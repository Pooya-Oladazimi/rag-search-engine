from actions.vars import DATASET
import json


def load_dataset():
    with open(DATASET, "r") as f:
        dataset = json.loads(f.read())
        return dataset["movies"]


def normalize_scores(numbers: list[float]):
    if len(numbers) == 0:
        return None
    minNum = min(numbers)
    maxNum = max(numbers)
    if minNum == maxNum:
        return [1.0] * len(numbers)

    scores = []
    for n in numbers:
        scores.append((n - minNum) / (maxNum - minNum))
    return scores
