import random
from pathlib import Path
import numpy as np
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio


class AcousticUnitsDataset(Dataset):
    def __init__(
        self,
        root: Path,
        sample_rate: int = 16000,
        min_samples: int = 32000,
        max_samples: int = 250000,
        train: bool = True,
    ):
        self.wavs_dir = root / "wav"
        self.units_dir = root / "discrete"
        self.f0_dir = root / "f0"
        self.phon_dir = root / "phon"

        # with open(root / "lengths.json") as file:
        #     self.lenghts = json.load(file)

        # pattern = "train-*/**/*.wav" if train else "dev-*/**/*.wav"
        pattern = "*.npy"
        metadata = (
            (path, path.relative_to(self.phon_dir).with_suffix("").as_posix())
            for path in self.phon_dir.rglob(pattern)
        )

        self.metadata = [
            path for path, key in metadata if len(np.load(path)) > min_samples
        ]
        print(self.metadata)

        self.sample_rate = sample_rate
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.train = train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        print("Test")
        f0_path = self.metadata[index]
        wav_path = self.wavs_dir / f0_path.relative_to(self.phon_dir)
        units_path = self.units_dir / f0_path.relative_to(self.phon_dir)
        phon_path = self.phon_dir / f0_path.relative_to(self.phon_dir)
        
        print(wav_path.with_suffix(".wav"))
        wav, _ = torchaudio.load(wav_path.with_suffix(".wav"))
        print(wav)
        codes = np.load(units_path)
        phon = torch.from_numpy(np.array(np.load(phon_path)))
        f0 = torch.from_numpy(np.array(np.load(f0_path))).float()

        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        phon = F.pad(phon, ((400 - 320) // 2, (400 - 320) // 2))
        f0 = F.pad(f0, ((400 - 320) // 2, (400 - 320) // 2))

        return wav, torch.from_numpy(codes).long(), phon, f0

    def collate(self, batch):
        wavs, codes, phons, f0 = zip(*batch)
        wavs, codes, phons, f0 = list(wavs), list(codes), list(phons), list(f0)

        wav_lengths = [wav.size(-1) for wav in wavs]
        code_lengths = [code.size(-1) for code in codes]
        phons_lengths = [phon.size(-1) for phon in phons]
        f0_lengths = [f.size(-1) for f in f0]

        phon_frames, phon_col, phon_off = self.modify_offsets(phons, phons_lengths)
        f0_frames, f0_col, f0_off = self.modify_offsets(f0, f0_lengths)
        wav_frames, wav_col, wav_off = self.modify_offsets(wavs, wav_lengths)

        rate = self.label_rate / self.sample_rate

        code_offsets = [round(wav_offset * rate) for wav_offset in wav_off]
        code_frames = round(wav_frames * rate)
        remaining_code_frames = [
            length - offset for length, offset in zip(code_lengths, code_offsets)
        ]
        code_frames = min(code_frames, *remaining_code_frames)

        collated_codes = []
        for code, code_offset in zip(codes, code_offsets):
            code = code[code_offset : code_offset + code_frames]
            collated_codes.append(code)

        wavs = torch.stack(wav_col, dim=0)
        codes = torch.stack(collated_codes, dim=0)
        phons = torch.stack(phon_col, dim=0)
        f0 = torch.stack(f0_col, dim=0)

        return wavs, codes, phons, f0
    
    def modify_offsets(self, batch, lengths):
        frames = min(self.max_samples, *lengths)

        collated_files, offsets = [], []
        for file in batch:
            diff = file.size(-1) - frames
            offset = random.randint(0, diff)
            file = file[:, offset : offset + frames]

            collated_files.append(file)
            offsets.append(offset)

        return frames, collated_files, offsets

