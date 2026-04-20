
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
import mne
from refshift.datasets.base import BaseDataset, CachedSubject
from refshift.preprocessing.filters import bandpass_filter_trials, resample_trials

SUPPORTED_EXTS = {'.vhdr','.edf','.bdf','.gdf','.set','.fif'}
LEFT_CODE = 769
RIGHT_CODE = 770


def _read_raw_any(path: Path):
    suf = path.suffix.lower()
    if suf == '.vhdr':
        return mne.io.read_raw_brainvision(str(path), preload=True, verbose='ERROR')
    if suf == '.edf':
        return mne.io.read_raw_edf(str(path), preload=True, verbose='ERROR')
    if suf == '.bdf':
        return mne.io.read_raw_bdf(str(path), preload=True, verbose='ERROR')
    if suf == '.gdf':
        return mne.io.read_raw_gdf(str(path), preload=True, verbose='ERROR')
    if suf == '.set':
        return mne.io.read_raw_eeglab(str(path), preload=True, verbose='ERROR')
    if suf == '.fif':
        return mne.io.read_raw_fif(str(path), preload=True, verbose='ERROR')
    raise ValueError(path)


def _subject_files(root: str, subject: int, include_runs: list[str]):
    sub_dir = Path(root) / f'sub-{subject:02d}' / 'eeg'
    out = []
    for p in sorted(sub_dir.glob('*_eeg.*')):
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        stem = p.name
        if any(r.lower() in stem.lower() for r in include_runs):
            out.append(p)
    return out


def _epoch_from_events(raw, events_tsv: Path, eeg_channels: list[str], tmin: float, tmax: float):
    df = pd.read_csv(events_tsv, sep='	')
    cue_df = df[df['trial_type'].isin([LEFT_CODE, RIGHT_CODE])].copy()
    data = raw.get_data(picks=eeg_channels).astype(np.float32)
    sfreq = float(raw.info['sfreq'])
    s0 = int(round(tmin * sfreq))
    s1 = int(round(tmax * sfreq))
    trials = []
    labels = []
    for _, row in cue_df.iterrows():
        start = int(row['sample']) + s0
        stop = int(row['sample']) + s1
        if start < 0 or stop > data.shape[1]:
            continue
        trials.append(data[:, start:stop])
        labels.append(0 if int(row['trial_type']) == LEFT_CODE else 1)
    if not trials:
        return None, None, sfreq
    return np.stack(trials, axis=0), np.array(labels, dtype=np.int64), sfreq

@dataclass
class Dreyer2023Dataset(BaseDataset):
    data_root: str
    spec: dict
    dataset_id: str = 'dreyer2023'
    def __post_init__(self):
        self.subject_list = list(self.spec['subjects_default'])
        self.channel_names = list(self.spec['native_channel_order'])
        self.include_runs = list(self.spec['run_policy']['include_runs'])
    def build_subject_cache(self, subject: int) -> CachedSubject:
        files = _subject_files(self.data_root, subject, self.include_runs)
        Xs, ys = [], []
        sfreq = None
        for eeg_file in files:
            raw = _read_raw_any(eeg_file)
            events_file = Path(str(eeg_file).replace('_eeg'+eeg_file.suffix, '_events.tsv'))
            X, y, sfreq = _epoch_from_events(raw, events_file, self.channel_names, *self.spec['window_sec_default'])
            if X is None:
                continue
            # scale EDF units V already? Keep as raw values from mne which should be volts.
            Xs.append(X.astype(np.float32))
            ys.append(y)
        if not Xs:
            raise RuntimeError(f'No usable acquisition epochs for Dreyer subject {subject}')
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        band = tuple(self.spec['bandpass_hz'])
        X = bandpass_filter_trials(X, sfreq, band)
        target = float(self.spec['target_sfreq_hz'])
        X = resample_trials(X, sfreq, target)
        return CachedSubject(self.dataset_id, subject, self.channel_names, target, 'random', X_all=X, y_all=y)
