
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import mne
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat
from refshift.datasets.base import BaseDataset, CachedSubject
from refshift.preprocessing.filters import bandpass_filter_trials, resample_trials


def _load_subject(path: str, tmin: float, tmax: float, eeg_channel_order: list[str]):
    data = loadmat(path, squeeze_me=True, struct_as_record=False, verify_compressed_data_integrity=False)['eeg']
    eeg_ch_names = list(eeg_channel_order)
    emg_ch_names = ['EMG1','EMG2','EMG3','EMG4']
    ch_names = eeg_ch_names + emg_ch_names + ['Stim']
    ch_types = ['eeg'] * len(eeg_ch_names) + ['emg'] * 4 + ['stim']
    imagery_left = data.imagery_left
    imagery_right = data.imagery_right
    eeg_data_l = np.vstack([imagery_left * 1e-6, data.imagery_event])
    eeg_data_r = np.vstack([imagery_right * 1e-6, data.imagery_event * 2])
    eeg_data = np.hstack([eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)), eeg_data_r])
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=float(data.srate))
    raw = RawArray(data=eeg_data, info=info, verbose=False)
    raw.set_montage(make_standard_montage('standard_1005'), on_missing='ignore')
    events = mne.find_events(raw, stim_channel='Stim', shortest_event=1, verbose=False)
    event_id = {'left_hand': 1, 'right_hand': 2}
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose='ERROR')
    epochs = epochs.copy().pick(eeg_ch_names)
    X = epochs.get_data().astype(np.float32, copy=False)
    inv = {v:k for k,v in epochs.event_id.items()}
    labs = np.array([inv[c] for c in epochs.events[:,2]])
    y = np.where(labs == 'left_hand', 0, 1).astype(np.int64)
    return X, y, float(data.srate)

@dataclass
class Cho2017Dataset(BaseDataset):
    data_root: str
    spec: dict
    dataset_id: str = 'cho2017'
    def __post_init__(self):
        self.subject_list = list(self.spec['subjects_default'])
        self.channel_names = list(self.spec['native_channel_order'])
    def build_subject_cache(self, subject: int) -> CachedSubject:
        path = str(Path(self.data_root) / f's{subject:02d}.mat')
        X, y, fs = _load_subject(path, *self.spec['window_sec_default'], self.channel_names)
        band = tuple(self.spec['bandpass_hz'])
        X = bandpass_filter_trials(X, fs, band)
        target = float(self.spec['target_sfreq_hz'])
        X = resample_trials(X, fs, target)
        return CachedSubject(self.dataset_id, subject, self.channel_names, target, 'random', X_all=X, y_all=y)
