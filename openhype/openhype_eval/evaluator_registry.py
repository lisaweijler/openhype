EVALUATOR_REGISTRY = {}


def register_evaluator(name):

    def decorator(cls):
        EVALUATOR_REGISTRY[name] = cls
        return cls

    return decorator


EVALUATOR_CONFIG_REGISTRY = {}


def register_evaluator_config(name):

    def decorator(cls):
        EVALUATOR_CONFIG_REGISTRY[name] = cls
        return cls

    return decorator
