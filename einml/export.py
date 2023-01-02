
## These are in their own file (instead of utils.py) so that they can be used by progress.py, which wants to
## be usable before importing tensorflow.

import sys

def export(item):

    if not hasattr(item, '__module__'):
        raise ValueError(f'Cannot @export {item}, it has no __module__ attribute. Use `exporter` and `export( ... )` instead')

    mod = sys.modules[item.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(item.__name__)
    else:
        mod.__all__ = [item.__name__]
    return item

def exporter():
    all = []
    def export(item):
        if isinstance(item, str):
            name = item
        elif hasattr(item, '__name__'):
            name = item.__name__
        else:
            raise ValueError(f'Cannot @export {item}, it has no `__name__` attribute. Use `export( ... )` instead')
        all.append(name)
        return item

    return all, export

export(export)
export(exporter)
