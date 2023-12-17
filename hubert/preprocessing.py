from pathlib import Path
import numpy as np
import torchaudio
import torch
from torchaudio import transforms
from tqdm import tqdm
import argparse
import pyworld as pw
from typing import Tuple

# Time conversion constants
# lab = 1*10^-7 seconds
LABTIME = 1e-7
# pyworld = 1*10^-4 seconds
# audio = 16000 samples per second

def convert_lab_to_samplerate(timing: int, sample_rate: int) -> float:
    return timing * LABTIME * sample_rate

def stretch_f0(f0: list, t: list, sample_rate: int, new_len: int) -> np.ndarray:
    """
    Stretch the F0 values to match a new length of audio.

    Parameters:
    f0 (list): List of F0 values.
    t (list): List of corresponding time values.
    sample_rate (int): Sample rate of the audio.
    new_len (int): New length of the audio.

    Returns:
    np.ndarray: Array of stretched F0 values.
    """

    # Create a new time array for the new length of audio
    new_t = np.arange(0, new_len / sample_rate, 1 / sample_rate)
    # Interpolate the F0 values for the new time array using linear interpolation
    new_f0 = np.interp(new_t, np.asarray(t), f0, left=np.nan, right=np.nan)

    return new_f0

def get_f0(wav: np.ndarray, sample_rate: int) -> Tuple[list[float], list[float]]:
    """
    Calculate the fundamental frequency (F0) of a given waveform.

    Args:
        wav (np.ndarray): The input waveform.
        sample_rate (int): The sample rate of the waveform.

    Returns:
        Tuple[list[float], list[float]]: A tuple containing the F0 values and corresponding time values.
    """
    _f0, t = pw.dio(wav, sample_rate)    # raw pitch extractor
    f0 = pw.stonemask(wav, _f0, t, sample_rate)  # pitch refinement
    return f0, t

def get_lab_array(lab_file: Path, sample_rate: int) -> Tuple[list[tuple[int, int, str]], list[str]]:
    """
    Reads a lab file and converts the timestamps to sample rate.
    
    Args:
        lab_file (Path): The path to the lab file.
        sample_rate (int): The sample rate of the audio.
    
    Returns:
        Tuple[list[tuple[int, int, str]], list[str]]: A tuple containing two lists.
            The first list contains tuples of start time, end time, and phoneme.
            The second list contains the phoneme corresponding to each sample of audio
    """
    with open(lab_file) as file:
        lines = file.readlines()
    lab_array = []
    phon_array = []
    for line in lines:
        start, end, phoneme = line.split()
        start = convert_lab_to_samplerate(int(start), sample_rate)
        end = convert_lab_to_samplerate(int(end), sample_rate)
        lab_array.append((start, end, phoneme))

        for i in range(int(start), int(end)):
            phon_array.append(phoneme)
    
    return lab_array, phon_array

def preprocess_file(wav_file: Path, lab_file: Path, sample_rate: int) -> Tuple[np.ndarray, list[str], np.ndarray]:
    """
    Preprocesses a WAV file and corresponding label file.

    Args:
        wav_file (Path): Path to the WAV file.
        lab_file (Path): Path to the label file.
        sample_rate (int): Desired sample rate for the preprocessed waveform.

    Returns:
        Tuple[np.ndarray, list[str], np.ndarray]: A tuple containing the preprocessed F0 array, phoneme array, and waveform.
    """
    
    # Load resample and adjust datatype
    waveform, og_sample_rate = torchaudio.load(wav_file)
    transform = transforms.Resample(og_sample_rate, sample_rate)
    waveform = transform(waveform).squeeze().to(torch.float64)

    # Make stereo file mono
    if waveform.size()[0] == 2:
        waveform = torch.mean(waveform, dim=0)
    waveform = waveform.numpy()

    # Get f0 and lab array
    f0, t = get_f0(waveform, sample_rate)
    lab_array, phon_array = get_lab_array(lab_file, sample_rate)
    f0 = stretch_f0(f0, t, sample_rate, len(waveform))
    return f0, phon_array, waveform

def preprocess_files(root_path: Path, output_path: Path, lab_folder: str, wav_folder: str, sample_rate: int) -> Tuple[np.ndarray, list[str]]:
    """
    Preprocesses audio and label files.

    Args:
        root_path (Path): The root path of the files.
        output_path (Path): The output path to save the preprocessed files.
        lab_folder (str): The folder name containing the label files.
        wav_folder (str): The folder name containing the audio files.
        sample_rate (int): The sample rate of the audio files.

    Returns:
        Tuple[np.ndarray, list[str]]: A tuple containing the preprocessed f0 array and phoneme list.
    """

    # Check if output folders exist, else create them
    if not (output_path / "f0").exists():
        (output_path / "f0").mkdir(parents=True)
    if not (output_path / "phon").exists():
        (output_path / "phon").mkdir(parents=True)
    if not (output_path / "wav").exists():
        (output_path / "wav").mkdir(parents=True)

    wav_path = root_path / wav_folder # Get path to wav files

    for wav_file in tqdm(list(wav_path.glob("*.wav")), desc="Preprocessing files"):
        lab_file = root_path / lab_folder / (wav_file.stem + ".lab") # Get corresponding lab file
        f0, phon_array, waveform = preprocess_file(wav_file, lab_file, sample_rate) # Preprocess file

        # Sometimes the phon_array is one element too short, in that case we duplicate the last element
        if len(phon_array) < waveform.shape[0]:
            for i in range(waveform.shape[0] - len(phon_array)):
                phon_array.append(phon_array[-1])

        assert(len(phon_array) == waveform.shape[0] and f0.shape[0] == waveform.shape[0])

        # Save files
        np.save(output_path / "f0" / (wav_file.stem + ".npy"), f0)
        np.save(output_path / "phon" / (wav_file.stem + ".npy"), phon_array)
        torchaudio.save(output_path / "wav" / (wav_file.stem + ".wav"), torch.from_numpy(waveform).unsqueeze(0), sample_rate)
    return f0, phon_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="data")
    parser.add_argument("--output_path", type=str, default="data")
    parser.add_argument("--lab_folder", type=str, default="lab")
    parser.add_argument("--wav_folder", type=str, default="wav")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()

    root_path = Path(args.root_path)
    output_path = Path(args.output_path)
    lab_folder = args.lab_folder
    wav_folder = args.wav_folder
    sample_rate = args.sample_rate
    preprocess_files(root_path, output_path, lab_folder, wav_folder, sample_rate)