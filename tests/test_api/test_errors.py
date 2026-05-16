from fastapi.testclient import TestClient


class TestErrorHandling:
    def test_unknown_route_returns_404(self, client: TestClient):
        response = client.get("/v1/nonexistent")
        assert response.status_code == 404

    def test_validation_error_structured(self, client: TestClient):
        response = client.get("/v1/predict/theater/book_00001")
        assert response.status_code == 422
        body = response.json()
        assert "request_id" in body
        assert body["error"] == "Validation Error"
        assert body["status_code"] == 422

    def test_response_has_request_id_header(self, client: TestClient):
        response = client.get("/v1/ready")
        assert "X-Request-ID" in response.headers
