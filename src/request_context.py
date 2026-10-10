
import uuid
from contextvars import ContextVar

request_id_context = ContextVar(
    "request_id",
    default=None
)


def new_request_id():
    return str(uuid.uuid4())


def get_request_id():
    return request_id_context.get()


def set_request_id(request_id=None):
    request_id = request_id or new_request_id()
    token = request_id_context.set(request_id)
    return request_id, token


def reset_request_id(token):
    request_id_context.reset(token)
