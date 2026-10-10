
from request_context import (
    get_request_id,
    new_request_id,
    reset_request_id,
    set_request_id,
)


def test_new_request_ids_are_unique():
    first = new_request_id()
    second = new_request_id()

    assert first != second


def test_request_id_is_set_and_reset():
    request_id, token = set_request_id("test-request-123")

    try:
        assert request_id == "test-request-123"
        assert get_request_id() == "test-request-123"
    finally:
        reset_request_id(token)

    assert get_request_id() is None


def test_request_id_is_generated_when_not_supplied():
    request_id, token = set_request_id()

    try:
        assert request_id
        assert get_request_id() == request_id
    finally:
        reset_request_id(token)
