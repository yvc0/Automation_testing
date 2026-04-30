"""
Vertex AI based RAG evaluation
"""

import json
import vertexai
from vertexai.preview.evaluation import EvalTask
from vertexai.preview.evaluation.metrics import (
    FaithfulnessMetric,
    AnswerRelevanceMetric,
    ContextRecallMetric
)
from config.config_loader import Config
from utils.report import save_json, save_html


def load_data():
    return json.load(open("data/rag_dataset.json"))


def transform(data):
    return [
        {
            "input": d["query"],
            "prediction": d["response"],
            "reference": d.get("reference", ""),
            "contexts": d["contexts"]
        }
        for d in data
    ]


def main():
    config = Config()

    vertexai.init(project=config.project_id, location=config.location)

    dataset = transform(load_data())

    task = EvalTask(
        dataset=dataset,
        metrics=[
            FaithfulnessMetric(),
            AnswerRelevanceMetric(),
            ContextRecallMetric(k=3)
        ]
    )

    result = task.evaluate()

    output = {
        "summary": result.summary_metrics,
        "details": result.metrics_table.to_dict("records")
    }

    save_json(output, "reports/rag.json")
    save_html(output["summary"], "reports/rag.html", "RAG Eval")


if __name__ == "__main__":
    main()
