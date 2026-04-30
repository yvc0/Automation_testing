import requests

class AgentClient:
    def __init__(self, endpoint, timeout):
        self.endpoint = endpoint
        self.timeout = timeout

    def call(self, user_input, session_id):
        res = requests.post(
            self.endpoint,
            json={"input": user_input, "session_id": session_id},
            timeout=self.timeout
        )
        data = res.json()

        return {
            "response": data.get("response", ""),
            "tool_calls": data.get("tool_calls", [])
        }
