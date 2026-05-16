from fastapi.testclient import TestClient


class TestHealthEndpoints:
    def test_readiness_returns_200(self, client: TestClient):
        response = client.get("/v1/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    def test_health_returns_200(self, client: TestClient):
        response = client.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in ("healthy", "degraded", "unhealthy")
        assert "uptime_seconds" in body
        assert "models_loaded" in body
        assert isinstance(body["models_loaded"], list)

    def test_health_includes_required_fields(self, client: TestClient):
        response = client.get("/v1/health")
        body = response.json()
        required = [
            "status", "uptime_seconds", "models_loaded",
            "last_batch_timestamp", "cache_connected",
            "feature_schema_valid", "prediction_count_total",
        ]
        for field in required:
            assert field in body, f"Missing field: {field}"

    def test_openapi_docs_available(self, client: TestClient):
        response = client.get("/v1/docs")
        assert response.status_code == 200

    def test_openapi_schema_available(self, client: TestClient):
        response = client.get("/v1/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "Cinema Audience Forecast API"
