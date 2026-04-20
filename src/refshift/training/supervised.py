
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np

from refshift.preprocessing.reference_ops import apply_reference
from refshift.preprocessing.standardization import standardize_trials_instance, fit_train_standardizer, apply_train_standardizer
from refshift.utils.seeding import set_all_seeds
from refshift.utils.metrics import classification_metrics
from refshift.models.atcnet import build_atcnet


@dataclass
class TrainConfig:
    seed: int = 1
    epochs: int = 200
    batch_size: int = 64
    lr: float = 3e-4
    standardization: str = 'mechanistic'
    n_windows: int = 5
    eegn_F1: int = 16
    eegn_D: int = 2
    eegn_kernel: int = 64
    eegn_pool: int = 8
    eegn_dropout: float = 0.3
    tcn_depth: int = 2
    tcn_kernel: int = 8
    tcn_filters: int = 32
    tcn_dropout: float = 0.3
    attention: str = 'mha'


def _prepare_fixed(Xtr, Xte, train_mode, test_mode, ch_names, neighbor_map, partner_map, std_mode):
    Xtr_ref = apply_reference(Xtr, train_mode, ch_names=ch_names, neighbor_map=neighbor_map, partner_map=partner_map)
    Xte_ref = apply_reference(Xte, test_mode, ch_names=ch_names, neighbor_map=neighbor_map, partner_map=partner_map)
    if std_mode == 'mechanistic':
        Xtr_ref = standardize_trials_instance(Xtr_ref)
        Xte_ref = standardize_trials_instance(Xte_ref)
    elif std_mode == 'deployment':
        mu, sd = fit_train_standardizer(Xtr_ref)
        Xtr_ref = apply_train_standardizer(Xtr_ref, mu, sd)
        Xte_ref = apply_train_standardizer(Xte_ref, mu, sd)
    else:
        raise ValueError(std_mode)
    return Xtr_ref[:, None, :, :], Xte_ref[:, None, :, :]


def fit_fixed_atcnet(Xtr, ytr, Xval, yval, n_classes, cfg: TrainConfig):
    set_all_seeds(cfg.seed)
    import tensorflow as tf
    model = build_atcnet(
        n_classes=n_classes,
        in_chans=Xtr.shape[2],
        in_samples=Xtr.shape[3],
        n_windows=cfg.n_windows,
        attention=cfg.attention,
        eegn_F1=cfg.eegn_F1,
        eegn_D=cfg.eegn_D,
        eegn_kernel=cfg.eegn_kernel,
        eegn_pool=cfg.eegn_pool,
        eegn_dropout=cfg.eegn_dropout,
        tcn_depth=cfg.tcn_depth,
        tcn_kernel=cfg.tcn_kernel,
        tcn_filters=cfg.tcn_filters,
        tcn_dropout=cfg.tcn_dropout,
        from_logits=False,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=0, callbacks=callbacks)
    return model


def predict_metrics(model, Xte, yte):
    prob = model.predict(Xte, verbose=0)
    pred = prob.argmax(axis=1)
    return classification_metrics(yte, pred), pred


class JitterSequence:
    def __init__(self, X, y, ch_names, neighbor_map, partner_map, train_modes, batch_size, seed, standardization='mechanistic', mixed_mu_sd=None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.ch_names = ch_names
        self.neighbor_map = neighbor_map
        self.partner_map = partner_map
        self.train_modes = list(train_modes)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.standardization = standardization
        self.mixed_mu_sd = mixed_mu_sd
        self.indices = np.arange(len(self.X))
        self.epoch = 0
    def __len__(self):
        return int(np.ceil(len(self.X)/self.batch_size))
    def on_epoch_end(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        rng.shuffle(self.indices)
        self.epoch += 1
    def __getitem__(self, idx):
        sl = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        Xb = self.X[sl]
        yb = self.y[sl]
        rng = np.random.default_rng(self.seed * 100003 + self.epoch * 1009 + idx)
        modes = rng.choice(self.train_modes, size=len(sl), replace=True)
        out = np.empty_like(Xb)
        for i, m in enumerate(modes):
            out[i] = apply_reference(Xb[i], str(m), ch_names=self.ch_names, neighbor_map=self.neighbor_map, partner_map=self.partner_map)
        if self.standardization == 'mechanistic':
            out = standardize_trials_instance(out)
        elif self.standardization == 'deployment':
            mu, sd = self.mixed_mu_sd
            out = apply_train_standardizer(out, mu, sd)
        else:
            raise ValueError(self.standardization)
        return out[:, None, :, :], yb


def fit_jitter_atcnet(Xtr_native, ytr, Xval_native, yval, n_classes, cfg: TrainConfig, ch_names, neighbor_map, partner_map, train_modes):
    set_all_seeds(cfg.seed)
    import tensorflow as tf
    if cfg.standardization == 'deployment':
        transformed = [apply_reference(Xtr_native, m, ch_names=ch_names, neighbor_map=neighbor_map, partner_map=partner_map) for m in train_modes]
        big = np.concatenate(transformed, axis=0)
        mixed_mu_sd = fit_train_standardizer(big)
        Xval_eval = {m: apply_train_standardizer(apply_reference(Xval_native, m, ch_names=ch_names, neighbor_map=neighbor_map, partner_map=partner_map), *mixed_mu_sd)[:, None,:,:] for m in train_modes}
    else:
        mixed_mu_sd = None
        Xval_eval = {m: standardize_trials_instance(apply_reference(Xval_native, m, ch_names=ch_names, neighbor_map=neighbor_map, partner_map=partner_map))[:, None,:,:] for m in train_modes}
    seq = JitterSequence(Xtr_native, ytr, ch_names, neighbor_map, partner_map, train_modes, cfg.batch_size, cfg.seed, cfg.standardization, mixed_mu_sd)
    model = build_atcnet(
        n_classes=n_classes,
        in_chans=Xtr_native.shape[1],
        in_samples=Xtr_native.shape[2],
        n_windows=cfg.n_windows,
        attention=cfg.attention,
        eegn_F1=cfg.eegn_F1,
        eegn_D=cfg.eegn_D,
        eegn_kernel=cfg.eegn_kernel,
        eegn_pool=cfg.eegn_pool,
        eegn_dropout=cfg.eegn_dropout,
        tcn_depth=cfg.tcn_depth,
        tcn_kernel=cfg.tcn_kernel,
        tcn_filters=cfg.tcn_filters,
        tcn_dropout=cfg.tcn_dropout,
        from_logits=False,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # use native validation transformed under first mode just to catch NaNs; final eval done separately
    first_mode = list(train_modes)[0]
    callbacks = [tf.keras.callbacks.TerminateOnNaN()]
    model.fit(seq, validation_data=(Xval_eval[first_mode], yval), epochs=cfg.epochs, verbose=0, callbacks=callbacks)
    return model
