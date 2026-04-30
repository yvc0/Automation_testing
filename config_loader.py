import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        env = os.getenv("ENV", "dev")
        load_dotenv(f".env.{env}")

        self.project_id = os.getenv("PROJECT_ID")
        self.location = os.getenv("LOCATION")

        self.agent_endpoint = os.getenv("AGENT_ENDPOINT")
        self.timeout = int(os.getenv("TIMEOUT", 20))
