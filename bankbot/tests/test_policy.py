from bankbot.app.core.language import LanguageNormalizer
from bankbot.app.core.policy import Policy
from bankbot.app.core.retrieval import Document, SearchResult


def test_policy_low_confidence_asks():
    normalizer = LanguageNormalizer()
    policy = Policy(normalizer)
    doc = Document(doc_id="kb_en_x", title="Test", content="Sample", tags=[], language="EN")
    results = [SearchResult(document=doc, score=0.05)]
    decision = policy.decide("test", "EN", results, threshold=0.2)
    assert decision.route == "ask"
    assert decision.clarifying_question


def test_policy_tool_call_emi():
    normalizer = LanguageNormalizer()
    policy = Policy(normalizer)
    decision = policy.decide("calculate emi P=500000 r=10 n=60", "EN", [])
    assert decision.route == "tool"
    assert "emi" in decision.tool_result
