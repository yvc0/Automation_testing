"""
Vertex AI pairwise evaluation (LLM judge)
"""

import json
import vertexai
from vertexai.preview.evaluation import EvalTask
from vertexai.preview.evaluation.metrics import PairwiseMetric
from config.config_loader import Config
from utils.report import save_json, save_html


def load_data():
    return json.load(open("data/autosxs_dataset.json"))


def transform(data):
    return [
        {
            "input": d["input"],
            "prediction": d["candidate_output"],
            "baseline_prediction": d["baseline_output"]
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
            PairwiseMetric(
                criteria=[
                    "helpfulness",
                    "instruction_following",
                    "coherence",
                    "fluency",
                    "safety"
                ]
            )
        ]
    )

    result = task.evaluate()

    output = {
        "summary": result.summary_metrics,
        "details": result.metrics_table.to_dict("records")
    }

    save_json(output, "reports/autosxs.json")
    save_html(output["summary"], "reports/autosxs.html", "Auto SxS Eval")


if __name__ == "__main__":
    main()
