import requests
from typing import Optional


class FedLedger:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://fed-ledger-prod.flock.io/api"
        self.api_version = "v1"
        self.url = f"{self.base_url}/{self.api_version}"
        self.headers = {
            "flock-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _get(self, endpoint: str):
        url = f"{self.url}{endpoint}"
        response = requests.get(url, headers=self.headers, timeout=30)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, json: Optional[dict] = None):
        url = f"{self.url}{endpoint}"
        return requests.post(url, headers=self.headers, json=json, timeout=30)

    def list_tasks(self):
        endpoint = "/tasks/list"
        return self._get(endpoint)

    def request_validation_assignment(self, task_id: str):
        endpoint = f"/tasks/request-validation-assignment/{task_id}"
        return self._post(endpoint)

    def submit_validation_result(self, assignment_id: str, data: dict):
        endpoint = f"/tasks/update-validation-assignment/{assignment_id}"
        data = {
            "status": "completed",
            "data": data,
        }
        return self._post(endpoint, json=data)

    def mark_assignment_as_failed(self, assignment_id: str):
        endpoint = f"/tasks/update-validation-assignment/{assignment_id}"
        data = {"status": "failed"}
        return self._post(endpoint, json=data)
