"""API endpoint tests for guided_learning router."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from deeptutor.api.routers.guided_learning import router
from deeptutor.learning.storage import LearningStore


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Create a minimal FastAPI app with only the guided_learning router.
    Monkeypatch LearningStore to use tmp_path for test isolation."""
    def _make_store_with_tmp(root=None):
        return LearningStore(root=tmp_path)
    monkeypatch.setattr(
        "deeptutor.api.routers.guided_learning.LearningStore",
        _make_store_with_tmp,
    )
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/learning")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# -- GET /progress (list_all) --------------------------------------------

class TestListProgress:
    def test_list_empty(self, client):
        resp = client.get("/api/v1/learning/progress")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_with_data(self, client):
        client.post("/api/v1/learning/progress/testbook/init-modules",
                    json={"modules": [{"id": "m1", "name": "M1", "order": 0,
                                       "knowledge_points": [{"id": "kp1", "name": "KP1",
                                                             "type": "concept", "module_id": "m1"}]}]})
        resp = client.get("/api/v1/learning/progress")
        assert resp.status_code == 200
        book_ids = [p["book_id"] for p in resp.json()]
        assert "testbook" in book_ids


# -- POST /progress/{book_id}/init-modules --------------------------------

class TestInitModules:
    def test_init_basic(self, client):
        resp = client.post("/api/v1/learning/progress/init1/init-modules",
                           json={"modules": [
                               {"id": "m1", "name": "Module 1", "order": 0,
                                "knowledge_points": [{"id": "kp1", "name": "KP1",
                                                      "type": "concept", "module_id": "m1"}]}
                           ]})
        assert resp.status_code == 200
        assert resp.json()["module_count"] == 1

    def test_init_empty_modules(self, client):
        resp = client.post("/api/v1/learning/progress/init2/init-modules",
                           json={"modules": []})
        assert resp.status_code == 200
        assert resp.json()["module_count"] == 0

    def test_init_invalid_kp_returns_422(self, client):
        resp = client.post("/api/v1/learning/progress/init3/init-modules",
                           json={"modules": [
                               {"id": "m1", "name": "M1", "order": 0,
                                "knowledge_points": [{"bad_key": "no_name"}]}
                           ]})
        assert resp.status_code == 422


# -- GET /progress/{book_id} ----------------------------------------------

class TestGetProgress:
    def test_get_progress_creates_on_fly(self, client):
        resp = client.get("/api/v1/learning/progress/newbook")
        assert resp.status_code == 200
        assert resp.json()["book_id"] == "newbook"

    def test_get_progress_invalid_id_returns_400(self, client):
        resp = client.get("/api/v1/learning/progress/a\\b")
        assert resp.status_code == 400


# -- DELETE /progress/{book_id} -------------------------------------------

class TestDeleteProgress:
    def test_delete_success(self, client):
        client.post("/api/v1/learning/progress/del1/init-modules",
                    json={"modules": []})
        resp = client.delete("/api/v1/learning/progress/del1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/api/v1/learning/progress/nonexistent42")
        assert resp.status_code == 404

    def test_delete_twice_returns_404(self, client):
        client.post("/api/v1/learning/progress/del2/init-modules",
                    json={"modules": []})
        client.delete("/api/v1/learning/progress/del2")
        resp = client.delete("/api/v1/learning/progress/del2")
        assert resp.status_code == 404

    def test_delete_invalid_book_id_returns_400(self, client):
        resp = client.delete("/api/v1/learning/progress/a\\b")
        assert resp.status_code == 400


# -- POST /progress/{book_id}/redo ----------------------------------------

class TestRedoProgress:
    def test_redo_resets_stage(self, client):
        client.post("/api/v1/learning/progress/redo1/init-modules",
                    json={"modules": [{"id": "m1", "name": "M1", "order": 0,
                                       "knowledge_points": []}]})
        resp = client.post("/api/v1/learning/progress/redo1/redo")
        assert resp.status_code == 200
        prog = client.get("/api/v1/learning/progress/redo1").json()
        assert prog["current_stage"] == "diagnostic_phase1"

    def test_redo_nonexistent_returns_404(self, client):
        resp = client.post("/api/v1/learning/progress/nope42/redo")
        assert resp.status_code == 404


# -- POST /progress/{book_id}/import-from-book ----------------------------

class TestImportFromBook:
    def test_import_two_chapters(self, client):
        resp = client.post("/api/v1/learning/progress/import1/import-from-book",
                           json={"chapters": [
                               {"title": "Ch1", "knowledge_points": ["KP1", "KP2"]},
                               {"title": "Ch2", "knowledge_points": ["KP3"]},
                           ]})
        assert resp.status_code == 200
        data = resp.json()
        assert data["module_count"] == 2
        assert data["status"] == "ok"

        prog = client.get("/api/v1/learning/progress/import1").json()
        assert len(prog["modules"]) == 2

    def test_import_empty_chapters(self, client):
        resp = client.post("/api/v1/learning/progress/import2/import-from-book",
                           json={"chapters": []})
        assert resp.status_code == 200
        assert resp.json()["module_count"] == 0


# -- POST /progress/{book_id}/generate-from-notebook ----------------------

class TestGenerateFromNotebook:
    def test_missing_records_returns_400(self, client):
        resp = client.post("/api/v1/learning/progress/nb1/generate-from-notebook",
                           json={"notebook_id": "nb", "records": []})
        assert resp.status_code == 400

    def test_invalid_book_id_returns_400(self, client):
        resp = client.post("/api/v1/learning/progress/a\\b/generate-from-notebook",
                           json={"notebook_id": "nb",
                                 "records": [{"id": "r1", "type": "note", "title": "T", "output": "O"}]})
        assert resp.status_code == 400


# -- book_id validation consistency ----------------------------------------

class TestBookIdValidation:
    """Verify all endpoints reject dangerous book_id characters."""

    # NOTE: `..` and `/` are normalized by HTTP clients before reaching the
    # handler, so they cannot be tested at the HTTP level.  Storage-level
    # path-traversal rejection is covered in test_storage.py.
    # Here we test `\` and `:` which survive URL transport.

    @pytest.mark.parametrize("method,path,body", [
        ("GET", "/api/v1/learning/progress/a\\b", None),
        ("DELETE", "/api/v1/learning/progress/a\\b", None),
        ("POST", "/api/v1/learning/progress/D:foo/init-modules", {"modules": []}),
        ("POST", "/api/v1/learning/progress/foo:bar/import-from-book", {"chapters": []}),
    ])
    def test_evil_book_id_rejected(self, client, method, path, body):
        kwargs = {"json": body} if body is not None else {}
        if method == "GET":
            resp = client.get(path, **kwargs)
        elif method == "POST":
            resp = client.post(path, **kwargs)
        elif method == "DELETE":
            resp = client.delete(path, **kwargs)
        assert resp.status_code == 400, f"{method} {path} should return 400, got {resp.status_code}"
