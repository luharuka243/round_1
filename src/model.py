from loguru import logger
from typing import Dict

class ModelClassifcation:
    def __init__(self, data: Dict):
        self.complaint_id = data["complaint_id"]
        self.content = data["content"]

    def predict(self):
        return {"category": "test", "sub_category": "test"}

    def process_data(self):
        pass
