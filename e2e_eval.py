"""
End-to-end conversation evaluation
"""

import json
import time
from config.config_loader import Config
from utils.agent_client import AgentClient
from utils.report import save_json, save_html


def load_data():
    return json.load(open("data/e2e_dataset.json"))


def evaluate(data, agent):
    success = 0
    latency = 0

    for item in data:
        start = time.time()
        final = ""

        for turn in item["turns"]:
            res = agent.call(turn["user"], item["session_id"])
            final = res["response"]

        latency += time.time() - start

        if any(x in final.lower() for x in item["success_criteria"]):
            success += 1

    total = len(data)

    return {
        "task_completion_rate": success / total,
        "avg_latency": latency / total
    }


def main():
    config = Config()
    agent = AgentClient(config.agent_endpoint, config.timeout)

    result = evaluate(load_data(), agent)

    save_json(result, "reports/e2e.json")
    save_html(result, "reports/e2e.html", "E2E Eval")


if __name__ == "__main__":
    main()
