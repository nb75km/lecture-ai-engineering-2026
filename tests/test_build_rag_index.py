"""実SQLiteと固定ベクトルで、索引作成から検索までを確認する。APIは呼ばない。"""

import json
from unittest.mock import Mock

import pytest

from app import retrieval
from scripts.build_rag_index import build_index


@pytest.fixture
def materials(tmp_path):
    source = tmp_path / "lectures.json"
    source.write_text(json.dumps([
        {"id": f"lec{n:02d}-main", "title": f"講義{n}", "content": [
            {"section": "章", "content": [
                {"id": f"lec{n:02d}-main-p001", "title": f"ページ{n}",
                 "page": 1, "content": f"本文{n}"}]}]}
        for n in (1, 2, 5)
    ], ensure_ascii=False), encoding="utf-8")
    return source, tmp_path / "index.sqlite3"


def test_built_index_can_be_searched(materials, monkeypatch):
    source, database = materials
    texts = []

    def embed(text, task):
        assert task == "RETRIEVAL_DOCUMENT"
        texts.append(text)
        return [1, 0] if "本文1" in text else [0, 1]

    result = build_index(source, database, embedder=embed)
    assert result["pages"] == 2  # 第5回の資料は検索対象に含めない
    assert all("章" in text and "ページ" in text for text in texts)
    monkeypatch.setattr(retrieval, "RAG_DB", database)
    monkeypatch.setattr(retrieval, "embed_text", lambda *_: [1, 0])
    hits = retrieval.search("質問")
    assert [h["page_id"] for h in hits] == ["lec01-main-p001", "lec02-main-p001"]
    assert hits[0]["content"] == "本文1"
    assert hits[0]["distance"] == pytest.approx(0)


def test_search_adds_only_adjacent_pages_in_the_same_document(materials, monkeypatch):
    source, database = materials
    documents = json.loads(source.read_text(encoding="utf-8"))
    for document, numbers in zip(documents, ([1, 2, 3, 4, 5, 7], [6], [1])):
        document["content"][0]["content"] = [
            {"id": f"{document['id']}-p{page:03}", "title": f"ページ{page}",
             "page": page, "content": f"本文{page}"}
            for page in numbers
        ]
    source.write_text(json.dumps(documents), encoding="utf-8")

    def embed_document(text, _task):
        return [1, 0] if text.splitlines()[-1] in {"本文1", "本文3", "本文5"} else [0, 1]

    build_index(source, database, embedder=embed_document)
    monkeypatch.setattr(retrieval, "RAG_DB", database)
    monkeypatch.setattr(retrieval, "RAG_TOP_K", 3)
    monkeypatch.setattr(retrieval, "embed_text", lambda *_: [1, 0])

    baseline = retrieval.search("質問")
    expanded = retrieval.search("質問", include_neighbors=True)

    assert [page["page"] for page in baseline] == [1, 3, 5]
    assert expanded[:3] == baseline
    assert [page["page_id"] for page in expanded[3:]] == ["lec01-main-p002", "lec01-main-p004"]
    assert [page["content"] for page in expanded[3:]] == ["本文2", "本文4"]
    assert [page["adjacent_to"] for page in expanded[3:]] == ["lec01-main-p001", "lec01-main-p003"]


def test_existing_index_survives_failed_rebuild(materials):
    source, database = materials
    build_index(source, database, embedder=lambda *_: [1, 0])
    before = database.read_bytes()
    embed = Mock(side_effect=RuntimeError("埋め込みに失敗"))
    with pytest.raises(FileExistsError):
        build_index(source, database, embedder=embed)
    embed.assert_not_called()
    with pytest.raises(RuntimeError, match="埋め込みに失敗"):
        build_index(source, database, rebuild=True, embedder=embed)
    assert database.read_bytes() == before
    assert not list(database.parent.glob(".rag-build-*"))
