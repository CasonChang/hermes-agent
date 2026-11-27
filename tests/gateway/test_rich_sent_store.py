from gateway.rich_sent_store import lookup, lookup_marker, record, record_marker


def test_text_and_markers_share_an_entry_scoped_to_the_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record("line:A", "1", "First bubble")
    record("line:B", "1", "Different conversation")
    record_marker("line:A", ["1", "2"], "line_own", "1")
    record("line:A", "2", "Second bubble")

    assert lookup("line:A", "1") == "First bubble"
    assert lookup_marker("line:A", "1", "line_own") == "1"
    assert lookup_marker("line:A", "2", "line_own") == "1"
    assert lookup("line:B", "1") == "Different conversation"
    assert lookup_marker("line:B", "1", "line_own") is None
