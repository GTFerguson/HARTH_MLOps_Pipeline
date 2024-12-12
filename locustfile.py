from locust import HttpUser, task

class ModelTestUser(HttpUser):
    @task
    def make_prediction(self):
        self.client.post("/invocations", json={
            "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        })
