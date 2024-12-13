from locust import HttpUser, TaskSet, task, between
import random

class FastAPITestTasks(TaskSet):
    @task
    def predict(self):
        """Simulate a POST request to the /predict endpoint."""
        payload = {
            "features": [
                random.uniform(-1, 1), 
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1)
            ]
        }
        with self.client.post("/predict", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code {response.status_code}: {response.text}")

class FastAPIUser(HttpUser):
    tasks = [FastAPITestTasks]
    wait_time = between(1, 3)  # Simulates a user waiting between 1 and 3 seconds between requests
