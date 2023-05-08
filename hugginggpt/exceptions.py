import functools


def wrap_exceptions(exception_cls, message=None):
    def decorated(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            try:
                return f(*args, **kwargs)
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
