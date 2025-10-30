from pathlib import Path

from bankbot.app.core.redaction import Redactor, detokenize


def test_redaction_replaces_pii():
    sample = "Call me at +919876543210 or email demo@example.com"
    redactor = Redactor()
    result = redactor.redact(sample)
    assert "+919876543210" not in result.text
    assert "demo@example.com" not in result.text
    assert len(result.mapping) == 2
    for token, original in result.mapping.items():
        assert detokenize(token, result.mapping) == original


def test_redaction_log_scrub(tmp_path: Path):
    from bankbot.app.core.redaction import write_audit_log

    log_path = tmp_path / "audit.log"
    redactor = Redactor()
    result = redactor.redact("Account 123456789012")
    write_audit_log(log_path, {"redaction_tokens": result.mapping.keys()})
    contents = log_path.read_text(encoding="utf-8")
    assert "123456789012" not in contents
