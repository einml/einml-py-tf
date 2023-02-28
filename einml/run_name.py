
import datetime
import os

from einml.export import export

@export
def random_run_name() -> str:
    import randomname
    return randomname.get_name()

@export
def get_run_name() -> str | None:
    try:
        return os.environ["RUN_NAME"]
    except KeyError:
        date = datetime.datetime.now().date().isoformat()
        return f"interactive-{date}"

@export
def set_run_name(run_name) -> str:
    os.environ["RUN_NAME"] = run_name
    return run_name
