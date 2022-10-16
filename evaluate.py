"""
A script to benchmark different VAD models on the AvaSpeech dataset. All VAD
models can be found in the `vads` directory of this project.

This script assumes that the audio files are downloaded in the
`dataset/ava_speech` folder, and normalized to be:
    - WAV
    - mono
    - 44.1k HZ

The AVA-Speech label CSV file should be located at the same directory as the
WAV files; and it has the following format:
--------------------------------------------
| audio_id | start_time | end_time | Label |
--------------------------------------------
"""
import os
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from utils import load_audio


class SpeechLabel:
    def __init__(self, start, end, label):
        self.start = start
        self.end = end
        self.label = label


def get_speech_boundaries(vad, audio, sr):
    boundaries = []
    # preprocess audio if needed
    audio, sr = vad._preprocess_audio(audio, sr)
    # get frames having speech
    frames = list(vad._split_to_frames(audio, sr))
    for speech_frame in vad._get_speech_frames(frames, sr):
        boundaries.append({
            "start": speech_frame["start"],
            "end": speech_frame["end"]
        })
    return boundaries


def write_labels(vad, audio_filepath, out_filepath):
    """Uses the given vad to get labels from the given audio."""
    audio, sr = load_audio(audio_filepath)
    audio_id = Path(audio_filepath).stem
    labels = [
        {
            "audio_id": audio_id,
            "start": b["start"],
            "end": b["end"],
            "label": "SPEECH"
        }
        for b in vad.get_speech_boundaries(audio, sr)
    ]
    df = pd.DataFrame(labels)
    df.to_csv(out_filepath, mode='a', header=False, index=False)


def parse_label_file(label_filepath, offset_time):
    """
    Parses a CSV file and returns a dictionary where the key is the audio's id
    and the value is a list of all SpeechLabels detected by the vad model.
    """
    headers = ["audio_id", "start_time", "end_time", "label"]
    df = pd.read_csv(label_filepath, names=headers)
    out = {}
    for audio_id, group in df.groupby('audio_id'):
        out[audio_id] = [
            SpeechLabel(
                start = row["start_time"]-offset_time,
                end = row["end_time"]-offset_time,
                label = row["label"]
            )
            for _, row in group.iterrows()
        ]
    return out


def get_duration_labels(true_labels, audio_ids):
    acc_true_labels = defaultdict(float)
    for audio_id in audio_ids:
        for true_label in true_labels[audio_id]:
            true_dur = true_label.end - true_label.start
            acc_true_labels[true_label.label] += true_dur
    return acc_true_labels


def get_overlaps(true_labels, hyp_labels):
    overlaps = defaultdict(float)
    for audio_id in tqdm(true_labels, "Calculating Overlap"):
        for true_label in true_labels[audio_id]:
            if audio_id in hyp_labels:
                for hyp_label in hyp_labels[audio_id]:
                    latest_start = max(true_label.start, hyp_label.start)
                    earliest_end = min(true_label.end, hyp_label.end)
                    dur = (
                        earliest_end - latest_start
                        if latest_start < earliest_end
                        else 0
                    )
                    overlaps[(true_label.label, hyp_label.label)] += dur
    return overlaps



def run(args):
    vads = __import__("vads")
    vad_models = [getattr(vads, cls)() for cls in args.vad_models]
    os.makedirs(args.out_path, exist_ok=True)
    
    audio_files = glob(str(args.dataset_path) + "/*.wav")
    # get labels for every vad model
    for vad in vad_models:
        vad_name = vad.__class__.__name__
        print(f"Using {vad_name} vad")
        # load predicted labels CSV file if already found
        out_filepath = os.path.join(args.out_path, vad_name+".csv")
        if os.path.exists(out_filepath):
            labeled_audio_ids = set(
                pd.read_csv(out_filepath).iloc[:,0].tolist()
            )
        # iterate over audio files
        for audio_filepath in tqdm(audio_files):
            audio_id = Path(audio_filepath).stem
            if audio_id in labeled_audio_ids:
                print(f"Audio {audio_id}.wav has been already labeled!")
            else:
                write_labels(vad, audio_filepath, out_filepath)
    
    # start evaluating
    label_filepath = os.path.join(args.dataset_path, "ava_speech_labels_v1.csv")
    true_labels = parse_label_file(label_filepath, offset_time=900)
    for vad_name in args.vad_models:
        out_label_filepath = os.path.join(args.out_path, vad_name+".csv")
        hyp_labels = parse_label_file(out_label_filepath, offset_time=0)
        dur_true_labels = get_duration_labels(true_labels, hyp_labels.keys())
        overlaps = get_overlaps(true_labels, hyp_labels)
        print(overlaps)
        print(dur_true_labels)
        print()
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", required=True, type=Path,
        help="Relative/Absolute path where AVA-Speech audio files are located."
    )
    parser.add_argument(
        "--out-path", default="labels", type=Path,
        help="Relative/Absolute path where the out labels will be located."
    )
    parser.add_argument(
        "--vad-models", nargs='*',
        help="List of vad models to be used.",
        choices=["WebRTC", "Silero", "SpeechBrain"],
    )
    parser.add_argument(
        "--show-confusion-matrix", action='store_true',
        help="A flag to print the confusion matrix for each vad model."
    )
    parser.add_argument(
        "--plot-PR-curve", action='store_true',
        help="A flag to plot the Precision-Recall Curve for all models."
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
