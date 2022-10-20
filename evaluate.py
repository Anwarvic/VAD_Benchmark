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
from itertools import product
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

from utils import load_audio


vad = None # GLOBAL VARIABLE


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
            {
                "start": row["start_time"]-offset_time,
                "end": row["end_time"]-offset_time,
                "label": row["label"]
            }
            for _, row in group.iterrows()
        ]
    return out


def get_speech_boundaries(audio_filepath):
    """Uses the given vad to get speech from the given audio."""
    global vad
    # read audio
    audio, sr = load_audio(audio_filepath)
    # preprocess audio if needed
    audio, sr = vad._preprocess_audio(audio, sr)
    # get frames having speech
    frames = list(vad._split_to_frames(audio, sr))
    speech_frames = vad._get_speech_frames(frames, audio, sr)
    return [
        {
            "start": speech_frame["start"] / sr, #in milliseconds 
            "end": speech_frame["end"] / sr, #in milliseconds
            "label": "SPEECH"
        }
        for speech_frame in speech_frames
    ]


def get_speech_boundaries_parallel(audio_filepaths, n_workers, desc=None):
    """Runs get_speech_boundaries() function in parallel."""
    boundaries = process_map(
        get_speech_boundaries,
        audio_filepaths,
        max_workers=n_workers,
        desc=desc
    )
    # sanity check
    assert len(boundaries) == len(audio_filepaths)
    # return labels
    out_speech_labels = {}
    for audio_filepath, speech_boundaries in zip(audio_filepaths, boundaries):
        audio_id = Path(audio_filepath).stem
        out_speech_labels[audio_id] = speech_boundaries
    return out_speech_labels


def get_precision_recall(
        segments_per_audio_ref,
        segments_per_audio_hyp,
        speech_labels
    ):
    """Gets precision and recall of VAD output labels."""
    speech_dur_ref = 0
    speech_dur_hyp = 0
    speech_overlap_dur = 0
    # iterate over processed audios
    for audio_id in tqdm(segments_per_audio_hyp, "Calculating P/R"):
        # cumulate ref. speech duration
        for ref_seg in segments_per_audio_ref[audio_id]:
            if ref_seg["label"] in speech_labels:
                speech_dur_ref += ref_seg["end"] - ref_seg["start"]
        # iterate over hypothesized speech segments
        for hyp_seg in segments_per_audio_hyp[audio_id]:
            # cumulate hyp. speech duration
            speech_dur_hyp += hyp_seg["end"] - hyp_seg["start"]
            # iterate over segments of true labels
            for ref_seg in segments_per_audio_ref[audio_id]:
                # ignore non-speech segments
                if ref_seg["label"] not in speech_labels:
                    continue
                # get overlap
                latest_start = max(ref_seg["start"], hyp_seg["start"])
                earliest_end = min(ref_seg["end"], hyp_seg["end"])
                speech_overlap_dur += (
                    earliest_end - latest_start
                    if latest_start < earliest_end
                    else 0
                )
    # calculate precision & recall
    P = speech_overlap_dur / speech_dur_hyp
    R = speech_overlap_dur / speech_dur_ref
    return P, R



def run(args):
    global vad
    vads = __import__("vads")
    os.makedirs(args.out_path, exist_ok=True)

    # parse AVA-Speech true labeled segments
    label_filepath = os.path.join(args.dataset_path, "ava_speech_labels_v1.csv")
    segments_per_audio_ref = (
        parse_label_file(label_filepath, offset_time=900)
    )
    
    # get available audio files
    audio_files = glob(str(args.dataset_path) + "/*.wav")
    # create plotting figure
    plt.figure()
    plt.xlabel("Recall"); plt.ylabel("Precision")
    for vad_name in args.vad_models:
        Ps, Rs =[], [] # precisions & recalls
        for agg, win_sz in product(args.agg_thresholds, args.window_sizes_ms):
            # initialize VAD model with current params
            vad = getattr(vads, vad_name)(
                threshold=agg,
                window_size_ms=win_sz
            )
            # get speech frames for every audio file
            segments_per_audio_hyp = get_speech_boundaries_parallel(
                audio_files,
                n_workers=args.num_workers,
                desc=f"VAD: {vad_name} (agg={agg}, win_sz={win_sz})"
            )
            # calculate precision & recall of the current VAD model
            P, R = get_precision_recall(
                segments_per_audio_ref,
                segments_per_audio_hyp,
                args.speech_labels
            )
            F1 = (2*P*R)/(P+R)
            print(f"Precision: {P}, Recall: {R}, F1: {F1}")
            Ps.append(P); Rs.append(R)
        # plot P/R curve
        plt.plot(Rs, Ps, label=vad_name)
    # save plot
    plt.savefig(args.out_path / "PR_curve.png")
    plt.legend(); plt.autoscale(); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", required=True, type=Path,
        help="Relative/Absolute path where AVA-Speech audio files are located."
    )
    parser.add_argument(
        "--speech-labels", nargs='*', type=str,
        default=["CLEAN_SPEECH", "SPEECH_WITH_MUSIC", "SPEECH_WITH_NOISE"],
        help="List (space separated) of the true labels (case-sensitive) " + \
            "that we are considering as 'speech'.",
    )
    parser.add_argument(
        "--vad-models", required=True, nargs='*',
        help="List of vad models to be used.",
        choices=["WebRTC", "Silero", "SpeechBrain"],
    )
    parser.add_argument(
        "--window-sizes-ms", required=True, nargs='*', type=int,
        help="List of window-sizes (in milliseconds) to be used.",
    )
    parser.add_argument(
        "--agg-thresholds", required=True, nargs='*', type=float,
        help="List of aggressiveness thresholds to be used. " + \
            "The higher the value is, the less sensitive the model gets.",
    )
    parser.add_argument(
        "--out-path", default=".", type=Path,
        help="Relative/Absolute path where the out labels will be located."
    )
    parser.add_argument(
        "--num-workers", type=int, default=os.cpu_count()-1,
        help="Number of workers working in parallel."
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
