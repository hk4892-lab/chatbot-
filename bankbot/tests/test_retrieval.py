from pathlib import Path

from bankbot.app.core.language import LanguageNormalizer
from bankbot.app.core.retrieval import KnowledgeBase, Retriever


def test_hinglish_retrieval(tmp_path: Path):
    data_path = Path(__file__).resolve().parents[1] / "app" / "data"
    kb = KnowledgeBase(data_path)
    normalizer = LanguageNormalizer()
    retriever = Retriever(kb)
    query = normalizer.normalize("UPI limit kitna hai")
    results = retriever.search(query)
    assert results
    assert results[0].document.doc_id == "kb_en_3"
    assert results[0].score > 0.18
