
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.io as sio
from refshift.datasets.base import BaseDataset, CachedSubject
from refshift.preprocessing.filters import bandpass_filter_trials, resample_trials


def _load_session_mat(path: str, subject: int, training: bool):
    name = f"A0{subject}{'T' if training else 'E'}.mat"
    return sio.loadmat(Path(path)/name)['data']


def _extract_trials(data_root: str, subject: int, training: bool, tmin: float, tmax: float, include_artifacts: bool = True):
    fs = 250.0
    win_len = 7 * int(fs)
    s0, s1 = int(round(tmin*fs)), int(round(tmax*fs))
    a_data = _load_session_mat(data_root, subject, training)
    X, y = [], []
    for ii in range(a_data.size):
        d = a_data[0, ii][0, 0]
        a_X, a_trial, a_y, a_art = d[0], d[1], d[2], d[5]
        for t in range(a_trial.size):
            if a_art[t] != 0 and not include_artifacts:
                continue
            seg = a_X[int(a_trial[t]) : int(a_trial[t]) + win_len, :22].T.astype(np.float32) * 1e-6
            X.append(seg[:, s0:s1])
            y.append(int(a_y[t]) - 1)
    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, fs

@dataclass
class IV2ADataset(BaseDataset):
    data_root: str
    spec: dict
    dataset_id: str = 'iv2a'
    def __post_init__(self):
        self.subject_list = list(self.spec['subjects_default'])
        self.channel_names = list(self.spec['native_channel_order'])
    def build_subject_cache(self, subject: int) -> CachedSubject:
        tmin, tmax = self.spec['window_sec_default']
        Xtr, ytr, fs = _extract_trials(self.data_root, subject, True, tmin, tmax, include_artifacts=True)
        Xte, yte, _ = _extract_trials(self.data_root, subject, False, tmin, tmax, include_artifacts=True)
        band = tuple(self.spec['bandpass_hz'])
        Xtr = bandpass_filter_trials(Xtr, fs, band)
        Xte = bandpass_filter_trials(Xte, fs, band)
        target = float(self.spec['target_sfreq_hz'])
        Xtr = resample_trials(Xtr, fs, target)
        Xte = resample_trials(Xte, fs, target)
        return CachedSubject(self.dataset_id, subject, self.channel_names, target, 'session', X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte)
