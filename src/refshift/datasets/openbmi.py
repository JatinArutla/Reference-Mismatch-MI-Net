
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from refshift.datasets.base import BaseDataset, CachedSubject
from refshift.preprocessing.filters import bandpass_filter_trials, resample_trials


def _mat_to_dict(mat_obj):
    if isinstance(mat_obj, np.ndarray) and mat_obj.dtype == object and mat_obj.size == 1:
        mat_obj = mat_obj.item()
    if isinstance(mat_obj, np.void):
        out = {}
        for name in mat_obj.dtype.names:
            out[name] = _mat_to_dict(mat_obj[name])
        return out
    if isinstance(mat_obj, np.ndarray) and mat_obj.dtype.names is not None:
        if mat_obj.size == 1:
            return _mat_to_dict(mat_obj.reshape(-1)[0])
        return [_mat_to_dict(x) for x in mat_obj]
    return mat_obj


def _extract_trials_from_openbmi_mat(mat_path: str):
    mat = loadmat(mat_path, simplify_cells=False)
    out = {}
    for key in ['EEG_MI_train','EEG_MI_test']:
        d = _mat_to_dict(mat[key])
        X = np.array(d['x'])
        y = np.array(d['y_dec']).squeeze().astype(np.int64) - 1
        # assume epoched [C,T,N] or [N,C,T]
        perms = [(0,1,2),(2,0,1),(2,1,0),(0,2,1),(1,0,2),(1,2,0)]
        best = None
        for p in perms:
            shape = [X.shape[i] for i in p]
            N,C,T = shape
            if C == 62 and N == len(y):
                best = p; break
        if best is None:
            raise RuntimeError(f'Could not infer OpenBMI tensor layout for {mat_path}: {X.shape}')
        Xn = np.transpose(X, best).astype(np.float32) * 1e-6
        out[key] = (Xn, y)
        channels = [str(c).strip() for c in np.array(d['chan'], dtype=object).reshape(-1).tolist()]
        sfreq = float(np.array(d['fs']).reshape(-1)[0])
    return out['EEG_MI_train'][0], out['EEG_MI_train'][1], out['EEG_MI_test'][0], out['EEG_MI_test'][1], channels, sfreq

@dataclass
class OpenBMIDataset(BaseDataset):
    data_root: str
    spec: dict
    dataset_id: str = 'openbmi'
    def __post_init__(self):
        self.subject_list = list(self.spec['subjects_default'])
        self.channel_names = list(self.spec['native_channel_order'])
    def _file(self, subject: int, session: int):
        return str(Path(self.data_root) / f'sess{session:02d}_subj{subject:02d}_EEG_MI.mat')
    def build_subject_cache(self, subject: int) -> CachedSubject:
        p1 = self._file(subject, 1)
        p2 = self._file(subject, 2)
        X1a, y1a, X1b, y1b, chans1, fs1 = _extract_trials_from_openbmi_mat(p1)
        X2a, y2a, X2b, y2b, chans2, fs2 = _extract_trials_from_openbmi_mat(p2)
        # each file contains train/test parts for that session; concatenate within session
        X1 = np.concatenate([X1a, X1b], axis=0); y1 = np.concatenate([y1a, y1b], axis=0)
        X2 = np.concatenate([X2a, X2b], axis=0); y2 = np.concatenate([y2a, y2b], axis=0)
        assert chans1 == chans2 == self.channel_names
        band = tuple(self.spec['bandpass_hz'])
        X1 = bandpass_filter_trials(X1, fs1, band)
        X2 = bandpass_filter_trials(X2, fs2, band)
        target = float(self.spec['target_sfreq_hz'])
        X1 = resample_trials(X1, fs1, target)
        X2 = resample_trials(X2, fs2, target)
        return CachedSubject(self.dataset_id, subject, self.channel_names, target, 'session', X_train=X1, y_train=y1, X_test=X2, y_test=y2)
