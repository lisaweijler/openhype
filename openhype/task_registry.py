TASK_REGISTRY = {}


def register_task(name):

    def decorator(func):
        TASK_REGISTRY[name] = func
        return func

    return decorator
