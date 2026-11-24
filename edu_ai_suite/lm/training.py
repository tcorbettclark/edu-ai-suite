"""One run's settings and results, with the optimizer loop that fills the weights.

The settings are one frozen object per kind of fitting, and each architecture names the
one it accepts: :class:`TrainingOptions` for the models an optimizer fits, and
:class:`CountingOptions` for the bigram, whose weights are counted. Both carry the
split, because every fit divides its pairs the same way.

The optimizer loop is :func:`fit_weights_by_optimizer`; sampling during training
happens in the caller's ``on_epoch`` hook, not here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import keras

from .pairs import Pairs
from .stats import EpochStats


@dataclass(frozen=True)
class SplitOptions:
    """What every fit needs: how the pairs are divided into training and validation."""

    validation_fraction: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be strictly between 0 and 1")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")


@dataclass(frozen=True, kw_only=True)
class TrainingOptions(SplitOptions):
    """How the optimizer runs: the settings of a model fitted by gradient descent."""

    epochs: int = 15
    batch_size: int = 256
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")


@dataclass(frozen=True, kw_only=True)
class CountingOptions(SplitOptions):
    """How counts become logits: the bigram's one setting, with no default.

    ``smoothing`` is the ``k`` of add-k smoothing. Each cell of the table starts with
    ``k`` invented observations, which is what stops a pair that never occurred from
    being impossible; ``1.0`` is Laplace's rule of succession, and smaller values suit a
    large vocabulary better.
    """

    smoothing: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.smoothing <= 0:
            raise ValueError(
                "smoothing must be > 0: a pair the corpus does not contain would get zero "
                "probability, and its loss would be infinite"
            )


@dataclass(frozen=True)
class TrainingResults:
    """The counts and final losses of one run, beyond the weights.

    Perplexity is ``e^loss``, derived from the loss rather than stored.
    """

    n_pairs_total: int
    n_train_pairs: int
    n_val_pairs: int
    final_loss: float
    final_val_loss: float

    @property
    def final_perplexity(self) -> float:
        return math.exp(self.final_loss)

    @property
    def final_val_perplexity(self) -> float:
        return math.exp(self.final_val_loss)


def compile_keras_model(
    keras_model: keras.Model, learning_rate: float | None = None
) -> keras.Model:
    """The same model with a loss attached, and an optimizer only if a rate is given.

    Sparse cross-entropy reads the raw logits, so the Keras graph carries no softmax. A
    counted model is never fitted by Keras, so it takes the loss alone - which is what
    ``evaluate`` reads.
    """
    keras_model.compile(
        optimizer=(
            keras.optimizers.Adam(learning_rate=learning_rate) if learning_rate is not None else None
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return keras_model


def loss_on(keras_model: keras.Model, pairs: Pairs, batch_size: int = 256) -> float:
    """The mean cross-entropy of the model over the pairs, in nats per prediction."""
    return float(
        keras_model.evaluate(pairs.contexts, pairs.targets, batch_size=batch_size, verbose=0)
    )


def fit_weights_by_optimizer(
    keras_model: keras.Model,
    train: Pairs,
    val: Pairs,
    options: TrainingOptions,
    *,
    on_epoch: Callable[[EpochStats], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> tuple[float, float]:
    """Compile the model, run its epochs over the pairs, and report the final losses.

    Returns the last epoch's ``(loss, val_loss)``. ``on_epoch`` receives one dict per
    epoch, and ``stop_requested`` is polled after every epoch and ends the Keras loop
    cleanly rather than raising.
    """
    compile_keras_model(keras_model, options.learning_rate)

    callbacks = []
    if on_epoch is not None or stop_requested is not None:
        callbacks.append(epoch_callback(on_epoch=on_epoch, stop_requested=stop_requested))

    history = keras_model.fit(
        train.contexts,
        train.targets,
        validation_data=(val.contexts, val.targets),
        epochs=options.epochs,
        batch_size=options.batch_size,
        callbacks=callbacks,
        verbose=0,
    ).history
    return float(history["loss"][-1]), float(history["val_loss"][-1])


def epoch_callback(
    *,
    on_epoch: Callable[[EpochStats], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> keras.callbacks.Callback:
    """A Keras callback forwarding one stats dict per epoch and polling the stop flag.

    ``on_epoch`` receives one dict per epoch. ``stop_requested`` is polled after every
    epoch and ends the Keras loop cleanly rather than raising.
    """

    def emit(epoch: int, logs: dict) -> None:
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        # Named, one argument per key: the dict's order is EpochStats' business
        # (and the checkpoint table's), never this call's.
        stats = EpochStats(
            epoch=epoch + 1,
            loss=float(loss) if loss is not None else float("nan"),
            val_loss=float(val_loss) if val_loss is not None else float("nan"),
            perplexity=math.exp(loss) if loss is not None else float("nan"),
            val_perplexity=math.exp(val_loss) if val_loss is not None else float("nan"),
        )
        on_epoch(stats)

    class _EpochCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
            if stop_requested is not None and stop_requested():
                self.model.stop_training = True  # clean Keras early stop
            if on_epoch is not None:
                emit(epoch, logs or {})

    return _EpochCallback()
