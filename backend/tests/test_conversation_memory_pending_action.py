import time


def test_pending_action_roundtrip_and_clear():
    from app.core.conversation_memory import ShortTermMemoryStore

    store = ShortTermMemoryStore(namespace="test-medrag", session_ttl_seconds=60, max_messages=5)
    session_id = "s-test-1"

    payload = {"type": "provider_disambiguation", "intent": "ask_price", "candidates": [1, 2, 3]}
    store.save_pending_action(session_id, payload)

    loaded = store.get_pending_action(session_id)
    assert loaded is not None
    assert loaded["type"] == "provider_disambiguation"
    assert loaded["intent"] == "ask_price"
    assert loaded["candidates"] == [1, 2, 3]

    store.clear_pending_action(session_id)
    assert store.get_pending_action(session_id) is None


def test_pending_action_expires_with_session_ttl():
    from app.core.conversation_memory import ShortTermMemoryStore

    store = ShortTermMemoryStore(namespace="test-medrag-ttl", session_ttl_seconds=1, max_messages=5)
    session_id = "s-test-ttl"

    store.save_pending_action(session_id, {"type": "provider_disambiguation"})
    assert store.get_pending_action(session_id) is not None

    time.sleep(1.2)
    assert store.get_pending_action(session_id) is None

