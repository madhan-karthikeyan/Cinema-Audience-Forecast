from fastapi.testclient import TestClient


class TestPredictionEndpoints:
    def test_batch_predict_pipeline_not_initialized(self, client: TestClient):
        response = client.post("/v1/predict/batch", json={})
        assert response.status_code in (503, 200)

    def test_batch_predict_rejects_invalid_alpha(self, client: TestClient):
        response = client.post("/v1/predict/batch", json={"blend_alpha": 1.5})
        assert response.status_code == 422

    def test_single_predict_pipeline_not_initialized(self, client: TestClient):
        response = client.get(
            "/v1/predict/theater/book_00001",
            params={"target_date": "2024-04-15"},
        )
        assert response.status_code in (503, 200)

    def test_single_predict_missing_date(self, client: TestClient):
        response = client.get("/v1/predict/theater/book_00001")
        assert response.status_code == 422

    def test_list_models(self, client: TestClient):
        response = client.get("/v1/models")
        assert response.status_code in (200, 501)

    def test_health_after_startup(self, client: TestClient):
        response = client.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert "prediction_count_total" in body
