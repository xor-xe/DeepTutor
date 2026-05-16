import tempfile
import time
from pathlib import Path

import pytest

from deeptutor.learning.models import KnowledgeType, LearningProgress, RepetitionState
from deeptutor.learning.storage import LearningStore


@pytest.fixture
def store(tmp_path):
    return LearningStore(root=tmp_path)


# ── save / load ──────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load(self, store):
        lp = LearningProgress(book_id="book1")
        lp.mastery_levels["kp1"] = 0.75
        store.save(lp)
        loaded = store.load("book1")
        assert loaded is not None
        assert loaded.book_id == "book1"
        assert loaded.mastery_levels["kp1"] == 0.75

    def test_enum_roundtrip(self, store):
        lp = LearningProgress(book_id="book1")
        lp.knowledge_types["kp1"] = KnowledgeType.MEMORY
        store.save(lp)
        loaded = store.load("book1")
        assert loaded.knowledge_types["kp1"] == KnowledgeType.MEMORY

    def test_repetition_state_roundtrip(self, store):
        lp = LearningProgress(book_id="book1")
        state = RepetitionState(interval_index=2, next_review_at=time.time() + 86400)
        lp.repetition_states["kp1"] = state
        store.save(lp)
        loaded = store.load("book1")
        assert loaded.repetition_states["kp1"].interval_index == 2

    def test_updated_at_auto_updates(self, store):
        lp = LearningProgress(book_id="book1")
        old_updated = lp.updated_at
        time.sleep(0.01)
        store.save(lp)
        loaded = store.load("book1")
        assert loaded.updated_at >= old_updated


# ── load nonexistent ─────────────────────────────────────────────────────

class TestLoadNonexistent:
    def test_returns_none(self, store):
        assert store.load("nonexistent") is None


# ── exists ───────────────────────────────────────────────────────────────

class TestExists:
    def test_true_after_save(self, store):
        store.save(LearningProgress(book_id="book1"))
        assert store.exists("book1") is True

    def test_false_when_missing(self, store):
        assert store.exists("nonexistent") is False


# ── delete ───────────────────────────────────────────────────────────────

class TestDelete:
    def test_removes_file(self, store):
        store.save(LearningProgress(book_id="book1"))
        store.delete("book1")
        assert store.load("book1") is None

    def test_delete_nonexistent_no_error(self, store):
        store.delete("nonexistent")  # should not raise


# ── path traversal ───────────────────────────────────────────────────────

class TestPathTraversal:
    def test_rejects_slash(self, store):
        with pytest.raises(ValueError, match="Invalid book_id"):
            store.load("../settings/foo")

    def test_rejects_backslash(self, store):
        with pytest.raises(ValueError, match="Invalid book_id"):
            store.load("a\\b")

    def test_rejects_dotdot(self, store):
        with pytest.raises(ValueError, match="Invalid book_id"):
            store.load("..")

    def test_rejects_colon(self, store):
        with pytest.raises(ValueError, match="Invalid book_id"):
            store.load("D:foo")

    def test_rejects_in_save(self, store):
        with pytest.raises(ValueError, match="Invalid book_id"):
            store.save(LearningProgress(book_id="../evil"))


# ── list_all ──────────────────────────────────────────────────────────────

class TestListAll:
    def test_list_all_empty(self, store):
        assert store.list_all() == []

    def test_list_all_multiple(self, store):
        store.save(LearningProgress(book_id="a"))
        store.save(LearningProgress(book_id="b"))
        ids = store.list_all()
        assert sorted(ids) == ["a", "b"]

    def test_list_all_after_delete(self, store):
        store.save(LearningProgress(book_id="x"))
        store.save(LearningProgress(book_id="y"))
        store.delete("x")
        assert store.list_all() == ["y"]
