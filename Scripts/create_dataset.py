"""
create_dataset.py

Preprocessing script for respiratory signals from polysomnography data.
Filters the raw signals to the breathing frequency range, chops them into
overlapping 30-second windows, and labels each window using the flow events
annotation file.

Usage:
    python create_dataset.py -in_dir "Data" -out_dir "Dataset"
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
import pickle
import warnings

warnings.filterwarnings("ignore")

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply a Butterworth bandpass filter to keep only breathing-relevant
    frequencies (roughly 0.17 – 0.4 Hz, i.e. 10–24 breaths per minute).

    A higher filter order gives a sharper roll-off but we keep it at 4
    to avoid ringing artefacts on short segments.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    low = max(low, 1e-4)
    high = min(high, 0.9999)

    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def load_signal(filepath):
    """
    Try to load a signal file. Handles .csv, .txt, .npy, and .mat formats.
    Returns (signal_array, sampling_frequency).
    """
    ext = os.path.splitext(filepath)[-1].lower()

    if ext in ('.csv', '.txt'):
        df = pd.read_csv(filepath)
        
        signal = df.iloc[:, 0].values.astype(float)
        if 'fs' in df.columns:
            fs = float(df['fs'].iloc[0])
        else:
            fs = 10.0  
        return signal, fs

    elif ext == '.npy':
        data = np.load(filepath, allow_pickle=True)
        
        if data.ndim == 2 and data.shape[0] == 2:
            return data[0].astype(float), float(data[1][0])
        return data.astype(float), 10.0

    elif ext == '.mat':
        mat = loadmat(filepath)
      
        keys = [k for k in mat.keys() if not k.startswith('__')]
        signal = mat[keys[0]].flatten().astype(float)
        fs = float(mat['fs'].flatten()[0]) if 'fs' in mat else 10.0
        return signal, fs

    else:
        raise ValueError(f"Unsupported file format: {ext}")


def load_annotations(filepath):
    """
    Load the flow-events annotation file.
    Expected columns: start_time (s), end_time (s), label (string).
    Returns a list of (start, end, label) tuples.
    """
    df = pd.read_csv(filepath)
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if 'start' in lc:
            col_map['start'] = col
        elif 'end' in lc or 'stop' in lc:
            col_map['end'] = col
        elif 'label' in lc or 'event' in lc or 'type' in lc:
            col_map['label'] = col

    if len(col_map) < 3:
        raise ValueError(
            f"Could not identify start/end/label columns in {filepath}. "
            f"Found columns: {list(df.columns)}"
        )

    events = []
    for _, row in df.iterrows():
        start = float(row[col_map['start']])
        end   = float(row[col_map['end']])
        label = str(row[col_map['label']]).strip()
        events.append((start, end, label))

    return events


def label_window(win_start, win_end, events):
    """
    Decide the label for a window that spans [win_start, win_end] seconds.

    Rule: if the window overlaps by more than 50% of its own duration with
    a labelled event, assign that event's label. Otherwise → 'Normal'.
    In the (unlikely) case of ties, the event with the greatest overlap wins.
    """
    win_dur = win_end - win_start
    best_label = 'Normal'
    best_overlap = 0.0

    for (ev_start, ev_end, ev_label) in events:
        overlap = max(0.0, min(win_end, ev_end) - max(win_start, ev_start))
        overlap_ratio = overlap / win_dur

        if overlap_ratio > 0.5 and overlap_ratio > best_overlap:
            best_overlap = overlap_ratio
            best_label = ev_label

    return best_label


def create_windows(signal, fs, events, window_sec=30, overlap=0.5):
    """
    Slide a window of `window_sec` seconds over the signal with `overlap`
    fractional overlap (0.5 = 50 %).

    Returns a list of dicts, each containing the window samples and its label.
    """
    win_samples  = int(window_sec * fs)
    step_samples = int(win_samples * (1 - overlap))

    windows = []
    start_idx = 0

    while start_idx + win_samples <= len(signal):
        end_idx = start_idx + win_samples

        win_start_sec = start_idx / fs
        win_end_sec   = end_idx   / fs

        segment = signal[start_idx:end_idx]
        lbl     = label_window(win_start_sec, win_end_sec, events)

        windows.append({
            'start_sec': win_start_sec,
            'end_sec':   win_end_sec,
            'label':     lbl,
            'signal':    segment
        })

        start_idx += step_samples

    return windows

def process_subject(signal_path, annotation_path, out_dir,
                    lowcut=0.17, highcut=0.4):
    """
    Full pipeline for one subject:
      1. Load signal + sampling rate
      2. Bandpass-filter to the breathing frequency band
      3. Slide windows with 50 % overlap
      4. Label each window from the annotation file
      5. Save to out_dir as both CSV (feature rows) and pickle (raw arrays)
    """
    subject_id = os.path.splitext(os.path.basename(signal_path))[0]
    print(f"\n  Processing: {subject_id}")


    signal, fs = load_signal(signal_path)
    print(f"     Signal length: {len(signal)} samples  |  fs = {fs} Hz  "
          f"({len(signal)/fs:.1f} s)")

    min_len = int(3 * fs / lowcut)
    if len(signal) < min_len:
        print(f"     WARNING: signal too short to filter reliably, skipping.")
        return

    filtered = bandpass_filter(signal, lowcut, highcut, fs)


    events  = load_annotations(annotation_path)
    windows = create_windows(filtered, fs, events)
    print(f"     Windows created: {len(windows)}")

    label_counts = {}
    for w in windows:
        label_counts[w['label']] = label_counts.get(w['label'], 0) + 1
    print(f"     Label distribution: {label_counts}")

    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for w in windows:
        row = {
            'subject':   subject_id,
            'start_sec': w['start_sec'],
            'end_sec':   w['end_sec'],
            'label':     w['label'],
            'signal':    ' '.join(map(str, w['signal']))
        }
        rows.append(row)

    csv_path = os.path.join(out_dir, f"{subject_id}_windows.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    pkl_path = os.path.join(out_dir, f"{subject_id}_windows.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(windows, f)

    print(f"     Saved → {csv_path}")
    print(f"     Saved → {pkl_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess respiratory signals into labelled windows."
    )
    parser.add_argument('-in_dir',  required=True,
                        help="Directory containing raw signal and annotation files.")
    parser.add_argument('-out_dir', required=True,
                        help="Directory where the processed dataset will be saved.")
    args = parser.parse_args()

    in_dir  = args.in_dir
    out_dir = args.out_dir

    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    os.makedirs(out_dir, exist_ok=True)
    signal_exts = {'.csv', '.txt', '.npy', '.mat'}
    signal_files = {}
    annot_files  = {}

    for fname in os.listdir(in_dir):
        fpath = os.path.join(in_dir, fname)
        base, ext = os.path.splitext(fname)

        if 'event' in base.lower() or 'annot' in base.lower():
            subject = base.lower().replace('_events', '').replace('_annot', '')
            annot_files[subject] = fpath
        elif ext.lower() in signal_exts:
            subject = base.lower().replace('_signal', '')
            signal_files[subject] = fpath

    matched = set(signal_files.keys()) & set(annot_files.keys())

    if not matched:
        print(
            "No matched signal+annotation pairs found.\n"
            "Expected naming: <id>_signal.<ext> and <id>_events.csv"
        )
        return

    print(f"Found {len(matched)} subject(s) to process: {sorted(matched)}")

    all_rows = []

    for subject in sorted(matched):
        try:
            process_subject(
                signal_path=signal_files[subject],
                annotation_path=annot_files[subject],
                out_dir=out_dir
            )
            csv_path = os.path.join(out_dir, f"{subject}_windows.csv")
            if os.path.exists(csv_path):
                all_rows.append(pd.read_csv(csv_path))

        except Exception as e:
            print(f"  ERROR processing {subject}: {e}")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined_path = os.path.join(out_dir, "dataset_all_subjects.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined dataset saved → {combined_path}")
        print(f"Total windows: {len(combined)}")
        print("Label counts:\n", combined['label'].value_counts().to_string())


if __name__ == '__main__':
    main()
