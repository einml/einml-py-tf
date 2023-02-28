from dataclasses import dataclass
import json
import math
from os import PathLike
from pathlib import Path
from threading import Thread
import time
import traceback
from typing import Callable, Literal, NamedTuple, Set, Union

from contextlib import ExitStack

from einml.prelude import *

from einml.progress import Progress, SubProgressManager, create_progress_manager
from einml.metrics import EinMetric, RunningMean, Rolling, TimeSinceLastCall, InstantaneousMetric
from einml.tf_types import NestedTensor

TrainStepReturn = tuple[tft.NestedTensor, tft.Tensor, list[tft.Tensor]]
TrainStepFn = Callable[[tft.NestedTensor, tft.NestedTensor], TrainStepReturn]

ValStepReturn = tuple[tft.NestedTensor, tft.Tensor]
ValStepFn = Callable[[tft.NestedTensor, tft.NestedTensor], ValStepReturn]

def default_make_train_step(
    model: Model,
    optimizer: keras.optimizers.Optimizer,
) -> TrainStepFn:
    """
    Factory function for default training step.
    """

    @tf.function
    @u.tf_scope
    def train_step(inputs, targets) -> TrainStepReturn:
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            loss = model.loss_fn(targets, outputs)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return outputs, loss, grads

    return train_step


def default_make_val_step(
    model: Model,
) -> ValStepFn:
    """
    Factory function for default validation step.
    """

    @tf.function
    @u.tf_scope
    def val_step(inputs, targets) -> ValStepReturn:
        outputs = model(inputs, training=False)
        loss = model.loss_fn(targets, outputs)
        return outputs, loss

    return val_step



def make_train_step_wrapper(
    model: Model,
    optimizer: tf.keras.optimizers.Optimizer,
    metrics: dict[str, EinMetric],
    make_train_step: Callable[..., TrainStepFn],
    tb_writer: tf.summary.SummaryWriter,
    do_log: bool,
    log_every: int,
    do_profile: bool,
) -> tft.GenericFunction:
    """
    Wraps a training step function. This is used to update metrics.

    Args:
    :param model: The model to train.
    :param data: The data to train on.
    """

    train_step = make_train_step(model, optimizer)

    @tf.function
    # @u.tf_scope
    def train_step_wrapper(
        data,
        step_var: tf.Variable,
        until_step: int,
        break_var: tf.Variable,
    ):
        for i_step, (inputs_batch, targets_batch) in data:

            with ExitStack() as stack:
                if do_profile:
                    stack.enter_context(tf.profiler.experimental.Trace("train", step_num=i_step, _r=1))

                outputs_batch, loss, grads = train_step(inputs_batch, targets_batch)

                step_var.assign(i_step)

                if break_var:
                    break

                if i_step >= until_step:
                    break

                metric_inputs: NestedTensor = {
                    "loss": loss,
                    "step": i_step,
                    "targets": targets_batch,
                    "outputs": outputs_batch,
                    "inputs": inputs_batch,
                }

                for metric in metrics.values():
                    metric.update(**metric_inputs)

                if do_log:
                    if i_step % log_every == 0:
                        with tb_writer.as_default(step=i_step):
                            tf.summary.scalar("loss", loss)
                            for grad in grads:
                                tf.summary.histogram("grad", grad)
                                tf.summary.scalar("grad_norm", tf.norm(grad))
                                if tf.is_tensor(outputs_batch):
                                    tf.summary.histogram(f"outputs", outputs_batch)
                                elif isinstance(outputs_batch, dict):
                                    for k, v in outputs_batch.items():
                                        tf.summary.histogram(f"outputs/{k}", v)
                                else:
                                    for i, v in enumerate(outputs_batch):
                                        tf.summary.histogram(f"outputs/{i}", v)

    return train_step_wrapper


def make_val_step_wrapper(
    model: Model,
    metrics: dict[str, EinMetric],
    make_val_step: Callable[..., ValStepFn] = default_make_val_step,
) -> tft.GenericFunction:
    """
    Wraps a validation step function.

    Args:
    :param model: The model to train.
    :param data: The data to train on.
    """

    val_step = make_val_step(model)

    @tf.function
    # @u.tf_scope
    def val_step_wrapper(
        data,
        step_var: tf.Variable,
        break_var: tf.Variable,
    ):
        for i_step, (inputs_batch, targets_batch) in data:

            outputs_batch, loss = val_step(inputs_batch, targets_batch)

            step_var.assign(i_step)

            if break_var:
                break

            metric_inputs: NestedTensor = {
                "loss": loss,
                "step": i_step,
                "targets": targets_batch,
                "outputs": outputs_batch,
                "inputs": inputs_batch,
            }

            for metric in metrics.values():
                metric.update(**metric_inputs)

    return val_step_wrapper



@export
def default_metrics(n_steps, n_val_steps) -> dict[str, dict[str, EinMetric]]:
    """
    Default metrics.
    """

    list_to_dict = lambda x: {item.name: item for item in x}
    get_loss = lambda **inputs: inputs["loss"]

    return {
        "epoch_metrics": list_to_dict([
            RunningMean(TimeSinceLastCall(), name="epoch_time", dtype=tf.float64, reset_every_epoch=False),
        ]),
        "train_step_metrics": list_to_dict([
            TimeSinceLastCall(name="step_time"),
            InstantaneousMetric(name="step", fn=lambda **inputs: inputs["step"], dtype=tf.int64, reset_every_epoch=False, fmt=f"{{}}/{n_steps}"),
            RunningMean(fn=get_loss, name="loss_epoch"),
            InstantaneousMetric(fn=get_loss, name="loss", reset_every_epoch=False),
            Rolling(length=100, fn=get_loss, name="loss_100", reset_every_epoch=False),
            Rolling(length=1000, fn=get_loss, name="loss_1000", reset_every_epoch=False),
        ]),
        "val_step_metrics": list_to_dict([
            RunningMean(fn=get_loss, name="eval_loss_epoch", reset_every_epoch=True),
            Rolling(length=n_val_steps, fn=get_loss, name="eval_loss", reset_every_epoch=False),
        ]),
    }

class TrainLoopError(Exception):

    def __init__(self, message, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = message


def get_epoch(i_step: int, epoch_sizes: list[int]):

    i_epoch = 0
    for epoch_size in epoch_sizes:
        if i_step < epoch_size:
            break
        i_step -= epoch_size
        i_epoch += 1

    return i_epoch

def get_remaining_steps_in_epoch(epoch_sizes, i_epoch, i_global_step):
    return epoch_sizes[i_epoch] - (i_global_step - sum(epoch_sizes[:i_epoch]))

@export
def make_train_loop(
    model: Model,
    data: Dataset,
    val_data: Dataset,
    output_dir: PathLike | str,
    optimizer: Literal["adam", "sgd"] | tf.keras.optimizers.Optimizer = "adam",
    n_steps_per_epoch: int | str = "funky",
    checkpoint_interval: Union[int, Literal["epoch"], Literal["never"]] = "epoch",
    log_interval: Union[int, Literal["epoch"], Literal["never"]] = 10,
    log_type: Set[Literal["tensorboard", "wandb"]] = {"tensorboard"},
    metrics: Union[
        Literal["default"],
        dict[str, dict[str, EinMetric]],
    ] = "default",
    make_train_step = default_make_train_step,
    make_val_step = default_make_val_step,
):
    """
    Make the training loop.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_steps = data.cardinality().numpy()
    n_val_steps = val_data.cardinality().numpy()

    # make epoch sizes
    if n_steps_per_epoch == "funky":
        # exponential, but not less than A or more than B
        epoch_sizes = [ e for e in u.funky_punky(20, 2000, n_steps)]
    elif n_steps_per_epoch == "exponential":
        epoch_sizes = [ e for e in u.exponential_up_to(n_steps) ]
    else:
        epoch_sizes = [ e for e in u.constant_up_to(n_steps, chunk=n_steps_per_epoch) ]

    # make optimizer
    if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
        if optimizer == "adam":
            # optimizer = tf.keras.optimizers.Adam(global_clipnorm=100., learning_rate=sched)
            optimizer = tf.keras.optimizers.Adam()
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer}")

    # make metrics
    if metrics == "default":
        metrics = default_metrics(n_steps=n_steps, n_val_steps=n_val_steps)
    elif isinstance(metrics, dict):
        assert "epoch_metrics" in metrics and "train_step_metrics" in metrics and "val_step_metrics" in metrics, "Metrics must be a dict with keys 'epoch_metrics', 'train_step_metrics', and 'val_step_metrics'."
    else:
        raise ValueError(f"Metrics was invalid. Must be a dict with keys 'epoch_metrics', 'train_step_metrics', and 'val_step_metrics'. Got: {metrics}")

    i, (inp, tar) = next(iter(data))
    model(inp)  # build the model
    if not hasattr(model, 'was_loaded') or not model.was_loaded:
        model.save(output_dir / model.name)

    # The following variables are immutable but are initialized
    # on the first call to train_loop. The boolean flags are
    # separate to support errors. The user can change the
    # implementations and use automatic reloading.
    created_initial_checkpoint = False
    created_data_iterator = True
    created_train_step = False
    created_val_step = False


    @u.tf_scope
    def train_loop(
        profile: bool = False,
        eager: bool = False,
        pm: Progress | None = None,
        log_type: Set[Literal["tensorboard", "wandb"]] = log_type,
        metrics: dict[str, dict[str, EinMetric]] = metrics,
    ):
        nonlocal created_initial_checkpoint, created_data_iterator, created_train_step, created_val_step, n_steps
        # The following variables are/have state.
        # They are here and not simply anonymously in the outer scope
        # so that they can be accessed by the caller
        if not hasattr(train_loop, "i_step"):
            train_loop.i_step = tf.Variable(
                0,
                dtype=tf.int64,
                trainable=False,
                name="i_step",
            )
        if not hasattr(train_loop, "i_val_step"):
            train_loop.i_val_step = tf.Variable(
                0,
                dtype=tf.int64,
                trainable=False,
                name="i_val_step",
            )
        if not hasattr(train_loop, "i_epoch"):
            train_loop.i_epoch = 0
        if not hasattr(train_loop, "checkpoints"):
            train_loop.checkpoints = []
        if not hasattr(train_loop, "optimizer"):
            train_loop.optimizer = optimizer
        if not hasattr(train_loop, "train_step_fn"):
            train_loop.train_step_fn = None

        if not hasattr(train_loop, "data_iterator"):
            train_loop.data_iterator = iter(data)

        if not hasattr(train_loop, "tb_writer"):
            if "tensorboard" in log_type:
                train_loop.tb_writer = tf.summary.create_file_writer(str(output_dir / "logs"))
            else:
                train_loop.tb_writer = None

        # run eagerly if requested
        # todo: currently this is not working
        tf.config.run_functions_eagerly(eager)

        with ExitStack() as stack:
            if pm is None:
                pm = stack.enter_context(create_progress_manager())

            if train_loop.tb_writer is not None:
                stack.enter_context(train_loop.tb_writer.as_default())

            def save_weights(final=False):
                if final:
                    suffix = "final"
                else:
                    suffix = f"step{train_loop.i_step.value().numpy()}"
                model.save_weights(output_dir / model.name / f"weights-{suffix}")

            try:

                try:
                    if profile:
                        tf.profiler.experimental.start(str(output_dir / "profile"))

                    train_loop_metrics = metrics["train_step_metrics"] | metrics["epoch_metrics"] | metrics["val_step_metrics"]


                    break_var = tf.Variable(
                        initial_value=False,
                        dtype=tf.bool,
                        trainable=False,
                        name="break_var",
                        # aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                        # synchronization=tf.VariableSynchronization.ON_WRITE,
                    )

                    # do_log: whether to enable logging in the train loop
                    # log_every: which steps to log on in the train loop
                    if log_interval == "epoch":
                        do_log = True
                        log_every = n_steps # only log once
                    elif log_interval == "never":
                        do_log = False
                        log_every = n_steps # ignored
                    elif isinstance(log_interval, int):
                        do_log = True
                        log_every = log_interval
                    else:
                        raise ValueError(f"Invalid log_interval: {log_interval}. Must be 'epoch', 'never', or an int.")

                    if not created_train_step:
                        with pm.enter_spinner(name="Compile train loop", desc="Compiling training loop..."):
                            step_fn = make_train_step_wrapper(
                                model,
                                optimizer,
                                metrics["train_step_metrics"],
                                make_train_step,
                                train_loop.tb_writer,
                                do_log=do_log,
                                log_every=log_every,
                                do_profile=profile,
                            )

                            train_loop.train_step_fn = step_fn.get_concrete_function(
                                data=train_loop.data_iterator,
                                step_var=train_loop.i_step,
                                until_step=train_loop.i_step + 1,
                                break_var=break_var,
                            )

                        created_train_step = True


                    if not created_val_step:
                        with pm.enter_spinner(name="Compile validation loop", desc="Compiling validation loop..."):
                            val_step_fn = make_val_step_wrapper(
                                model,
                                metrics["val_step_metrics"],
                                make_val_step,
                            )

                            train_loop.val_data_iterator = iter(val_data)

                            train_loop.val_step_fn = val_step_fn.get_concrete_function(
                                data=train_loop.val_data_iterator,
                                step_var=train_loop.i_step,
                                break_var=break_var,
                            )

                        created_val_step = True

                    ######################
                    ######################
                    @u.tf_scope
                    def run_for(sub_pm: SubProgressManager, n_steps, is_last_epoch):
                        nonlocal created_train_step

                        last_epoch_loss = None
                        with sub_pm.enter_progbar(total=n_steps, name=f"Epoch {train_loop.i_epoch + 1}", desc=f"Epoch {train_loop.i_epoch + 1}", delete_on_success=not is_last_epoch) as (sub_sub_pm, epoch_prog_bar):

                            exc = None
                            def run_in_thread():
                                nonlocal exc
                                try:
                                    train_loop.train_step_fn(
                                        data=train_loop.data_iterator,
                                        step_var=train_loop.i_step,
                                        until_step=train_loop.i_step + n_steps,
                                        break_var=break_var,
                                    )
                                except Exception as e:
                                    exc = e
                                    return
                            t = Thread(target=run_in_thread)

                            start_step = train_loop.i_step.value().numpy()

                            try:
                                t.start()

                                while t.is_alive():
                                    tstep = train_loop.i_step.value().numpy()
                                    train_loop.i_epoch = get_epoch(tstep, epoch_sizes)
                                    epoch_prog_bar.count = tstep - start_step
                                    time.sleep(0.01)

                            except KeyboardInterrupt as e:
                                break_var.assign(True)
                                t.join()
                                raise e
                            finally:
                                t.join()
                                if exc is not None:
                                    raise exc





                        with sub_pm.enter_progbar(total=n_val_steps, name="Validation", desc="Validating...", delete_on_success=not is_last_epoch) as (sub_sub_pm, val_prog_bar):

                            exc = None
                            def run_val_in_thread():
                                nonlocal exc
                                if train_loop.val_data_iterator is None:
                                    train_loop.val_data_iterator = iter(val_data)
                                try:
                                    train_loop.val_step_fn(
                                        data=train_loop.val_data_iterator,
                                        step_var=train_loop.i_val_step,
                                        break_var=break_var,
                                    )
                                    if not break_var.value().numpy():
                                        train_loop.val_data_iterator = None
                                except Exception as e:
                                    exc = e
                                    return

                            train_loop.i_val_step.assign(0)
                            t = Thread(target=run_val_in_thread)
                            try:
                                t.start()

                                while t.is_alive():
                                    vstep = train_loop.i_val_step.value().numpy()
                                    val_prog_bar.count = vstep
                                    time.sleep(0.01)

                            except KeyboardInterrupt as e:
                                break_var.assign(True)
                                t.join()
                                raise e
                            finally:
                                t.join()
                                if exc is not None:
                                    raise exc

                        val_loss = metrics["val_step_metrics"]["eval_loss"].result()

                        if not hasattr(train_loop, "best_val_loss") or val_loss < train_loop.best_val_loss:
                            train_loop.best_val_loss = val_loss
                            train_loop.best_val_epoch = train_loop.i_epoch
                            train_loop.best_val_step = train_loop.i_step.value().numpy()
                            train_loop.best_val_weights = model.get_weights()

                            with sub_pm.enter_spinner("Checkpoint", "Checkpointing weights...", delete_on_success=True):
                                save_weights()


                        for _, m in metrics["epoch_metrics"].items():
                            m.update(**{"epoch": train_loop.i_epoch, "step": train_loop.i_step})

                        if "loss_epoch" in train_loop_metrics:
                            epoch_loss_metric = train_loop_metrics["loss_epoch"]
                            epoch_loss = epoch_loss_metric.result()
                            if last_epoch_loss is not None:
                                if epoch_loss > last_epoch_loss:
                                    raise TrainLoopError("WARNING: Epoch *training* loss increased from last epoch. This is probably a bug in your model. Exiting training loop.")
                            last_epoch_loss = epoch_loss

                        # reset metrics
                        for _, m in train_loop_metrics.items():
                            if m.reset_every_epoch:
                                m.reset()

                    ######################
                    ######################

                    if not hasattr(train_loop, "start_time"):
                        train_loop.start_time = time.time()

                    with pm.enter_training(len(epoch_sizes), train_loop_metrics) as (sub_pm, train_prog_bar):
                        print(f"Starting training loop at step {train_loop.i_step.numpy()} (epoch {train_loop.i_epoch + 1})")

                        while train_loop.i_epoch < len(epoch_sizes):
                            print(f"epoch={train_loop.i_epoch}, step={u.tf_str(train_loop.i_step)}, epoch_size={epoch_sizes[train_loop.i_epoch]}")
                            step = train_loop.i_step.value().numpy()
                            n_steps = get_remaining_steps_in_epoch(epoch_sizes, train_loop.i_epoch, step)
                            if n_steps == 0:
                                raise TrainLoopError("WARNING: No steps left. Exiting training loop.")
                            run_for(sub_pm, n_steps, train_loop.i_epoch == (len(epoch_sizes) - 1))
                            step = train_loop.i_step.value().numpy()
                            train_loop.i_epoch = get_epoch(step + 1, epoch_sizes)
                            train_prog_bar.count = train_loop.i_epoch
                            train_prog_bar.refresh()
                finally:
                    if profile:
                        with pm.enter_spinner("Save profiler data", "Saving data from performance profiling..."):
                            tf.profiler.experimental.stop(save=True)

                print("Training loop finished.")

                json.dump({
                    "time_taken_s": time.time() - train_loop.start_time,
                    "n_steps": int(train_loop.i_step),
                    "model_name": model.name,
                }, open(output_dir / "details.json", "w"))


            except TrainLoopError as e:
                print(e.message)
            except KeyboardInterrupt:
                print("User interrupted training.")

            if hasattr(train_loop, "best_val_weights"):
                with pm.enter_spinner("Checkpoint", "Checkpointing weights...", delete_on_success=True):
                    model.set_weights(train_loop.best_val_weights)
                    save_weights(final=True)


    return train_loop
