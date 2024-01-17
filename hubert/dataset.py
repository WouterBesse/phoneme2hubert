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
        label_rate: int = 50,
        min_samples: int = 16000,
        max_samples: int = 20000,
        train: bool = True,
    ):
        self.wavs_dir = root / "wav"
        self.units_dir = root / "discrete"
        self.f0_dir = root / "f0"
        self.phon_dir = root / "phon"
        self.label_rate = label_rate

        # with open(root / "lengths.json") as file:
        #     self.lenghts = json.load(file)

        pattern = "train/*.npy" if train else "dev/*.npy"

        # pattern = "*.npy"
        metadata = (
            (path, path.relative_to(self.phon_dir).with_suffix("").as_posix())
            for path in self.phon_dir.rglob(pattern)
        )

        self.metadata = [
            path for path, key in metadata if len(np.load(path)) > min_samples
        ]
        # print(self.metadata)

        self.sample_rate = sample_rate
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.train = train

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        f0_path = self.metadata[index]
        units_path = self.units_dir / f0_path.relative_to(self.phon_dir)
        phon_path = self.phon_dir / f0_path.relative_to(self.phon_dir)
        
        codes = np.load(units_path)
        phon = torch.from_numpy(np.array(np.load(phon_path))).unsqueeze(0)
        f0 = torch.from_numpy(np.array(np.load(f0_path))).float().unsqueeze(0)
        # print(f0.size())
        f0_phon = torch.cat([f0, phon], dim=0)

        # wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        phon = F.pad(phon, ((400 - 320) // 2, (400 - 320) // 2))
        f0 = F.pad(f0, ((400 - 320) // 2, (400 - 320) // 2))

        return torch.from_numpy(codes), f0_phon

    def collate(self, batch):
        codes, f0_phons = zip(*batch)
        codes, f0_phons = list(codes), list(f0_phons)

        code_lengths = [code.size(-1) for code in codes]
        f0_phons_lengths = [f0_phon.size(-1) for f0_phon in f0_phons]

        f0_phon_frames, f0_phon_col, f0_phon_off = self.modify_offsets(f0_phons, f0_phons_lengths)
        # f0_frames, f0_col, f0_off = self.modify_offsets(f0, f0_lengths)

        rate = self.label_rate / self.sample_rate

        code_offsets = [round(wav_offset * rate) for wav_offset in f0_phon_off]
        code_frames = round(f0_phon_frames * rate)

        remaining_code_frames = [
            length - offset for length, offset in zip(code_lengths, code_offsets)
        ]
        code_frames = min(code_frames, *remaining_code_frames)

        collated_codes = []
        for code, code_offset in zip(codes, code_offsets):
            code = code[code_offset : code_offset + code_frames]
            collated_codes.append(code)

        codes = torch.stack(collated_codes, dim=0)
        f0_phons = torch.stack(f0_phon_col, dim=0)
        # f0 = torch.stack(f0_col, dim=0)

        # f0_phon = torch.cat([f0, phons], dim=1)

        return f0_phons, codes
    
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

