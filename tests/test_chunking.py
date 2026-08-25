from backend.app.services.chunking_service import ChunkingService


def test_empty_text_returns_no_chunks():
    service = ChunkingService()

    assert service.split("") == []


def test_short_text_returns_one_chunk():
    service = ChunkingService(
        chunk_size=100,
        overlap=20,
    )

    chunks = service.split("Hello enterprise support")

    assert len(chunks) == 1
    assert chunks[0] == "Hello enterprise support"


def test_long_text_creates_multiple_chunks():
    service = ChunkingService(
        chunk_size=100,
        overlap=20,
    )

    text = "A" * 500

    chunks = service.split(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)