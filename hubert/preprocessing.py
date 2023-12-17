from pathlib import Path
import numpy as np
import torchaudio
import torch
from torchaudio import transforms
from tqdm import tqdm
import argparse
import pyworld as pw

# lab = 1*10^-7 seconds
# pyworld = 1*10^-4 seconds
# audio = 16000 samples per second

def convert_lab_to_samplerate(timing: int, sample_rate: int):
    return timing * 1e-7 * sample_rate

def stretch_f0(f0: list, t: list, sample_rate: int, new_len: int):
    new_t = np.arange(0, new_len / sample_rate, 1 / sample_rate)
    # print(f0.shape, np.asarray(t).shape, new_t.shape)
    # print(t)
    # Interpolate the F0 values for the new time array using linear interpolation
    new_f0 = np.interp(new_t, np.asarray(t), f0, left=np.nan, right=np.nan)

    return new_f0

def get_f0(wav, sample_rate: int):
    _f0, t = pw.dio(wav, sample_rate)    # raw pitch extractor
    f0 = pw.stonemask(wav, _f0, t, sample_rate)  # pitch refinement
    return f0, t

def get_lab_array(lab_file: Path, sample_rate: int):
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

def preprocess_file(wav_file: Path, lab_file: Path, sample_rate: int):
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

def preprocess_files(root_path: Path, output_path: Path, lab_folder: str, wav_folder: str, sample_rate: int):
    # Check if output folders exist, else create them
    if not (output_path / "f0").exists():
        (output_path / "f0").mkdir(parents=True)
    if not (output_path / "phon").exists():
        (output_path / "phon").mkdir(parents=True)
    if not (output_path / "wav").exists():
        (output_path / "wav").mkdir(parents=True)

    wav_path = root_path / wav_folder
    for wav_file in tqdm(list(wav_path.glob("*.wav")), desc="Preprocessing files"):
        lab_file = root_path / lab_folder / (wav_file.stem + ".lab")
        f0, phon_array, waveform = preprocess_file(wav_file, lab_file, sample_rate)
        np.save(output_path / "f0" / (wav_file.stem + ".npy"), f0)
        np.save(output_path / "phon" / (wav_file.stem + ".npy"), phon_array)
        torchaudio.save(output_path / "wav" / (wav_file.stem + ".wav"), torch.from_numpy(waveform).unsqueeze(0), sample_rate)
        # np.save(output_path / "wav" / wav_file.with_suffix(".wav"), waveform)
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