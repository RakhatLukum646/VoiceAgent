from app.observability.errors import safe_error_fields


def test_safe_error_fields_omit_uri_and_secret():
    try:
        raise RuntimeError("failed wss://example.test/?key=secret-token")
    except RuntimeError as exc:
        fields = safe_error_fields(exc)
    assert fields["error_code"] == "runtime_error"
    assert fields["error_type"] == "RuntimeError"
    assert fields["trace"]
    assert "secret-token" not in str(fields)
    assert "wss://" not in str(fields)
