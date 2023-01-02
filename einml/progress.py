from __future__ import annotations
from contextlib import contextmanager
from math import isnan
from icecream import ic
import sys
import threading
import time
import traceback
from typing import Callable

import enlighten

from einml.export import export
from einml.run_name import get_run_name, set_run_name

## Note: we don't import using einml.prelude or einml.utils because we want this file
## to be usable before tensorflow is imported.

SPINNER_1X2 = "⠁⠂⠄⡀⢀⠠⠐⠈"
SPINNER_2X2 = ["⠁ ", "⠂ ", "⠄ ", "⡀ ", "⢀ ", " ⡀", " ⢀", " ⠠", " ⠐", " ⠈", " ⠁", "⠈ "]
SPINNER = SPINNER_2X2
CROSS = "⭕"
TICK = "💚"

VERTICAL_BAR = '│'
HORIZONTAL_BAR = '─'
LEFT_CORNER = "╭"
TOP_TEE = "┬"
LEFT_STOP = "╼"
RIGHT_STOP = "╾"
RIGHT_CORNER = "╮"


def metric_bar_format(metrics: dict[str, einml.metrics.EinMetric]):
    metrics = metrics.values()

    if len(metrics) == 0:
        empty_format = f"{VERTICAL_BAR} {{fill}}"
        return empty_format, empty_format

    headers = [(f"{m.name}", f" ({m.unit})" if m.unit else "") for m in metrics]
    metric_formats = [f"{{metric_{m.name}}}" for m in metrics]
    header_bar_format = f"{VERTICAL_BAR}{{fill}}{LEFT_CORNER}{f'{TOP_TEE}'.join(metric_formats)}{RIGHT_CORNER}{{fill}}"
    value_bar_format = f"{VERTICAL_BAR}{{fill}}{VERTICAL_BAR}{f'{VERTICAL_BAR}'.join(metric_formats)}{VERTICAL_BAR}{{fill}}"
    return header_bar_format, value_bar_format

def metric_format(metrics_dict: dict[str, EinMetric]):
    metrics = metrics_dict.values()

    if len(metrics) == 0:
        return {}, {}

    def value_format(m: EinMetric, width):

        val = m.result()

        width_nopad = width - 2
        if val is None:
            s = f"{' ... ':^{width_nopad}}"
        elif m.fmt is not None:
            s = m.fmt.format(val)
        elif isnan(val):
            s = f"{' NaN ':^{width_nopad}}"
        elif isinstance(val, int):
            s = f"{val:> {width_nopad}}"
        elif isinstance(val, str):
            s = f"{val:<{width_nopad}}"
        else: # assume float or float-like
            s = f"{val:> {width_nopad}.3g}"

        return f" {s} "

    def header_format(title, unit, width):
        return f"{ LEFT_STOP + ' ' + title + unit + ' ' + RIGHT_STOP :{HORIZONTAL_BAR}^{width}}"

    len_of_stoppers_and_gap = 4
    max_len_of_numeric_val = 13
    headers = [(f"{m.name}", f" ({m.unit})" if m.unit else "") for m in metrics]
    widths = [max(len(title) + len(unit) + len_of_stoppers_and_gap, max_len_of_numeric_val) for title, unit in headers]

    headers = [(f"{m.name}", f" ({m.unit})" if m.unit else "") for m in metrics]
    metric_names = [f"metric_{m.name}" for m in metrics]

    return {
        name: header_format(title, unit, width) for name, width, (title, unit) in zip(metric_names, widths, headers)
    }, {
        name: value_format(m, width) for name, width, m in zip(metric_names, widths, metrics)
    }


@export
class Progress:

    def __init__(
        self,
        manager: enlighten.Manager,
        min_delta: float = 0.5 / 8,
    ):
        self.manager = manager
        self.min_delta = min_delta
        self.tasks: list[str] = []
        self.update_fns: list[Callable[[int], None]] = []
        self.update_fns_to_remove = []

        title_format = LEFT_CORNER + "{fill}{task_format}{fill}" + LEFT_STOP
        self.title_bar=self.manager.status_bar(status_format=title_format, task_format=self.task_format(), fill=HORIZONTAL_BAR)

    def task_format(self):

        run_name = get_run_name()

        if len(self.tasks) == 0:
            return f" {run_name} "
        else:
            task_str = " > ".join(self.tasks)
            return f" {run_name}: {task_str} "

    @contextmanager
    def enter_spinner(
        self,
        name: str,
        desc: str,
        delete_on_success=False,
        indent_level: int = 0,
    ):
        indent = "    " * indent_level

        status_format = VERTICAL_BAR + " {indent}{spinner} {desc}"

        spinner_bar: enlighten.StatusBar = self.manager.status_bar(
            manager=self.manager,
            status_format=status_format,
            desc=desc,
            spinner=SPINNER[0],
            indent=indent,
            min_delta=self.min_delta,
            leave=True,
        )

        state = "running"
        closed = False
        def update(i):
            if not closed:
                if state == "running":
                    spinner = SPINNER[i % len(SPINNER)]
                elif state == "success":
                    spinner = TICK
                else:
                    spinner = CROSS
                spinner_bar.update(spinner=spinner, desc=desc)

        try:
            self.tasks.append(name)

            self.update_fns.append(update)

            sub_pm = SubProgressManager(self, indent_level=indent_level+1)

            yield sub_pm, spinner_bar

            if delete_on_success:
                closed = True
                spinner_bar.leave = False
                spinner_bar.close()
            else:
                desc += " done."
                state = "success"
        except Exception:
            state = "error"
            raise
        finally:
            if update in self.update_fns:
                def close():
                    if not closed:
                        spinner_bar.close()
                self.update_fns_to_remove.append((update, close))
            if len(self.tasks) > 0 and self.tasks[-1] == name:
                self.tasks.pop()

    @contextmanager
    def enter_progbar(
        self,
        total: int | None,
        name: str,
        desc: str,
        unit: str ='steps',
        start_at: int = 0,
        delete_on_success: bool = True,
        indent_level: int = 0
    ):
        indent = "    " * indent_level

        counter_format = VERTICAL_BAR + ' {indent}{spinner} {desc}{desc_pad}{count:d}{unit}{unit_pad}{fill}[ {elapsed}, {rate:.2f}{unit_pad}{unit}/s]'
        bar_format = VERTICAL_BAR + ' {indent}{spinner} {desc}{desc_pad}{percentage:3.0f}% |{bar}| {count:{len_total}d}/{total:d} {unit} [ {elapsed}<{eta}, {rate:.2f}{unit_pad}{unit}/s ]'

        prog_bar: enlighten.Counter = self.manager.counter(
            total=total,
            spinner=SPINNER[0],
            desc=desc,
            indent=indent,
            unit=unit,
            min_delta=self.min_delta,
            bar_format=bar_format,
            counter_format=counter_format,
            count=start_at,
            leave=True,
        )

        state = "running"
        closed = False
        def update(i):
            if not closed:
                if state == "running":
                    spinner = SPINNER[i % len(SPINNER)]
                elif state == "success":
                    spinner = TICK
                else:
                    spinner = CROSS
                prog_bar.update(incr=0, spinner=spinner)

        try:
            self.tasks.append(name)

            self.update_fns.append(update)

            sub_pm = SubProgressManager(self, indent_level=indent_level+1)

            yield sub_pm, prog_bar

            if delete_on_success:
                closed = True
                prog_bar.leave = False
                prog_bar.close()
            else:
                state = "success"
        except Exception:
            state = "error"
            raise
        finally:
            if update in self.update_fns:
                def close():
                    if not closed:
                        prog_bar.close()
                self.update_fns_to_remove.append((update, close))
            if len(self.tasks) > 0 and self.tasks[-1] == name:
                self.tasks.pop()

    @contextmanager
    def enter_training(self, n_epochs: int, metrics: dict[str, EinMetric] = {}, indent_level: int = 0):
        name = "Train"
        metrics_update = None

        header_fmt, value_fmt = metric_bar_format(metrics=metrics)
        headers, values = metric_format(metrics)

        try:
            metric_header_bar: enlighten.StatusBar = self.manager.status_bar(
                manager=self.manager,
                status_format=header_fmt,
                min_delta=self.min_delta,
                leave=True,
                **headers,
            )
            metric_value_bar: enlighten.StatusBar = self.manager.status_bar(
                manager=self.manager,
                status_format=value_fmt,
                min_delta=self.min_delta,
                leave=True,
                **values,
            )

            def metrics_update(i_step):
                headers, values = metric_format(metrics)
                metric_header_bar.update(**headers)
                metric_value_bar.update(**values)

            self.update_fns.append(metrics_update)

            with self.enter_progbar(total=n_epochs, name=name, desc="Overall Progress", unit="epochs", delete_on_success=False) as (sub_pm, prog_bar):

                yield sub_pm, prog_bar

        finally:
            if metrics_update is not None and metrics_update in self.update_fns:
                def close():
                    metric_header_bar.close()
                    metric_value_bar.close()
                self.update_fns_to_remove.append((metrics_update, close))

class SubProgressManager:

    def __init__(self, pm: Progress, indent_level: int):
        self.pm = pm
        self.indent_level = indent_level

    def enter_spinner(self, name: str, desc: str, delete_on_success: bool = False):
        return self.pm.enter_spinner(name, desc, delete_on_success=delete_on_success, indent_level=self.indent_level)

    def enter_progbar(self, total: int | None, name: str, desc: str, unit: str ='steps', start_at: int = 0, delete_on_success: bool = True):
        return self.pm.enter_progbar(total, name, desc, unit, start_at, delete_on_success, indent_level=self.indent_level)

    def enter_training(self, n_epochs: int, metrics: dict[str, EinMetric] = {}):
        return self.pm.enter_training(n_epochs, metrics, indent_level=self.indent_level)


@export
@contextmanager
def create_progress_manager():
    t = None
    update_title_bar = None
    with enlighten.get_manager() as e_manager:
        try:
            manager = Progress(
                manager=e_manager,
            )

            done = False
            def update():
                nonlocal done
                i = 0
                while not done:
                    l = len(manager.update_fns)
                    for j in range(l):
                        manager.update_fns[j](i)

                    for (f, close) in manager.update_fns_to_remove:
                        if f in manager.update_fns:
                            manager.update_fns.remove(f)
                            close()
                    i += 1
                    time.sleep(manager.min_delta)

            def update_title_bar(i):
                manager.title_bar.update(task_format=manager.task_format())

            manager.update_fns.append(update_title_bar)

            t = threading.Thread(None, update).start()

            yield manager


        except Exception:
            # traceback.print_exc()
            raise
        finally:
            sys.stderr.flush()
            sys.stdout.flush()
            time.sleep(manager.min_delta)
            done = True
            if t is not None:
                t.join()
            if update_title_bar is not None and update_title_bar in manager.update_fns:
                manager.update_fns_to_remove.append((update_title_bar, lambda: manager.title_bar.close()))

if __name__ == '__main__':
    set_run_name("Test")
    with create_progress_manager() as pm:
        with pm.enter_spinner("Loading", "Loading Data"):
            time.sleep(2)
        with pm.enter_training(3) as (pm2, prog_bar):
            for i in prog_bar(range(3)):
                with pm2.enter_progbar(3, f"Epoch {i}", f"Epoch {i}") as (pm3, epoch_bar):
                    for j in epoch_bar(range(3)):
                        time.sleep(0.1)
        with pm.enter_progbar(10, "Visualize", "Visualizing Data") as (sub_pm, prog_bar):
            for i in prog_bar(range(10)):
                time.sleep(0.1)

def init_with_progress():
    with create_progress_manager() as pm:
        with pm.enter_spinner("Init Tensorflow", "Initializing Tensorflow..."):
            import einml.prelude
