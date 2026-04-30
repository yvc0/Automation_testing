"""
Evaluates tool usage correctness (agent reasoning).
"""

import json
from collections import Counter
from config.config_loader import Config
from utils.agent_client import AgentClient
from utils.report import save_json, save_html


def load_data():
    return json.load(open("data/trajectory_dataset.json"))


def is_in_order(expected, actual):
    idx = 0
    for a in actual:
        if idx < len(expected) and a == expected[idx]:
            idx += 1
    return idx == len(expected)


def precision_recall(expected, actual):
    exp, act = Counter(expected), Counter(actual)
    common = sum((exp & act).values())
    p = common / len(actual) if actual else 0
    r = common / len(expected) if expected else 0
    return p, r


def evaluate(data, agent):
    details = []
    summary = {
        "trajectory_exact_match": 0,
        "trajectory_in_order_match": 0,
        "trajectory_any_order_match": 0,
        "trajectory_precision": 0,
        "trajectory_recall": 0
    }

    for item in data:
        expected = [t["name"] for t in item["expected_tools"]]
        actual = []

        for turn in item["turns"]:
            res = agent.call(turn, item["session_id"])
            actual += [t["name"] for t in res["tool_calls"]]

        exact = actual == expected
        inorder = is_in_order(expected, actual)
        anyorder = Counter(actual) == Counter(expected)
        p, r = precision_recall(expected, actual)

        details.append({
            "expected": expected,
            "actual": actual,
            "exact": exact,
            "in_order": inorder
        })

        summary["trajectory_exact_match"] += exact
        summary["trajectory_in_order_match"] += inorder
        summary["trajectory_any_order_match"] += anyorder
        summary["trajectory_precision"] += p
        summary["trajectory_recall"] += r

    total = len(data)
    for k in summary:
        summary[k] /= total

    return {"summary": summary, "details": details}


def main():
    config = Config()
    agent = AgentClient(config.agent_endpoint, config.timeout)

    result = evaluate(load_data(), agent)

    save_json(result, "reports/trajectory.json")
    save_html(result["summary"], "reports/trajectory.html", "Trajectory Eval")


if __name__ == "__main__":
    main()
