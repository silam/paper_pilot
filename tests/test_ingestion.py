from app.ingestion.chunking import chunk_document
from app.ingestion.cleaning import clean_documents, clean_text, redact_pii
from app.ingestion.loaders import RawDocument, parse_arxiv_atom


def _doc(text, i="d1"):
    print("*********** _doc")
    return RawDocument(doc_id=i, title="T", text=text, source="file")


def test_chunks_respect_size_and_carry_metadata():
    print("*********** test_chunks_respect_size_and_carry_metadata")
    doc = _doc(("Sentence number one is here. " * 200).strip())
    chunks = chunk_document(doc, size=300, overlap=50)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 + 50 + len("Title: T\n") + 5 for c in chunks)
    assert chunks[0].metadata["doc_id"] == "d1" and chunks[1].position == 1


def test_sentence_strategy_never_cuts_mid_sentence():
    print("*********** test_sentence_strategy_never_cuts_mid_sentence")
    doc = _doc("Alpha beta gamma. " * 100)
    for c in chunk_document(doc, size=200, overlap=0, strategy="sentence"):
        assert c.text.rstrip().endswith(".")


def test_clean_fixes_hyphenation_and_whitespace():
    print("*********** test_clean_fixes_hyphenation_and_whitespace")
    assert clean_text("trans-\nformer    model") == "transformer model"


def test_pii_redaction():
    print("*********** test_pii_redaction")
    text, counts = redact_pii("mail me at jane.doe@corp.com or 612-555-0199")
    assert "[EMAIL]" in text and "[PHONE]" in text and counts == {"EMAIL": 1, "PHONE": 1}


def test_dedup_and_short_docs_dropped():
    print("*********** test_dedup_and_short_docs_dropped")
    long = "A meaningful paragraph about transformers and attention. " * 3
    docs, report = clean_documents([_doc(long, "a"), _doc(long, "b"), _doc("tiny", "c")])
    assert len(docs) == 1 and report["duplicates"] == 1 and report["too_short"] == 1


def test_parse_arxiv_atom():
    print("*********** test_parse_arxiv_atom")
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/abs/2401.00001v1</id><title>A  Paper</title>
      <summary>Abstract text.</summary><published>2024-01-01T00:00:00Z</published>
      <author><name>Ada</name></author></entry></feed>"""
    [d] = parse_arxiv_atom(xml)
    assert d.doc_id == "arxiv:2401.00001v1" and d.title == "A Paper" and d.authors == ["Ada"]


def test_ingestion_is_idempotent(indexed):
    print("*********** test_ingestion_is_idempotent")
    from app.ingestion.loaders import load_file
    from app.ingestion.pipeline import SAMPLE, ingest_documents
    ingest_documents(load_file(SAMPLE))           # make sure everything is indexed
    again = ingest_documents(load_file(SAMPLE))   # second run must be a no-op
    assert again["new"] == 0 and again["updated"] == 0 and again["chunks"] == 0
