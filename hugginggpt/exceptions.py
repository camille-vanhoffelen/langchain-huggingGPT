import functools


def wrap_exceptions(exception_cls, message=None):
    """Wrap exceptions raised by a function with a custom exception class."""
    def decorated(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                raise exception_cls(message) from e

        return wrapped

    return decorated


def async_wrap_exceptions(exception_cls, message=None):
    """Wrap exceptions raised by an async function with a custom exception class."""
    def decorated(f):
        @functools.wraps(f)
        async def wrapped(*args, **kwargs):
            try:
                return await f(*args, **kwargs)
            except Exception as e:
                raise exception_cls(message) from e

        return wrapped

    return decorated


class TaskPlanningException(Exception):
    pass


class TaskParsingException(Exception):
    pass


class ModelScrapingException(Exception):
    pass


class ModelSelectionException(Exception):
    pass


class ModelInferenceException(Exception):
    pass


class ResponseGenerationException(Exception):
    pass
