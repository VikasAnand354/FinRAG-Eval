import json

def load_examples(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def dummy_pipeline(question: str):
    return {"answer": "dummy answer", "citations": [], "latency": 0.1}

def evaluate_example(example, prediction):
    correct = example["gold_answer"].lower() in prediction["answer"].lower()
    return {"example_id": example["example_id"], "correct": correct, "latency": prediction["latency"]}

def run_eval(dataset_path):
    examples = load_examples(dataset_path)
    results = []
    for ex in examples:
        pred = dummy_pipeline(ex["question"])
        results.append(evaluate_example(ex, pred))
    return results

if __name__ == "__main__":
    results = run_eval("data/splits/test.jsonl")
    print("Total:", len(results))
    print("Accuracy:", sum(r["correct"] for r in results) / len(results))
