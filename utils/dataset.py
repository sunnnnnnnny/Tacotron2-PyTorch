import os
import re
import torch
import pickle
import numpy as np
from text import text_to_sequence
from hparams import hparams as hps
from torch.utils.data import Dataset
from utils.audio import load_wav, melspectrogram


def files_to_list(fdir):
    f_list = []
    with open(os.path.join(fdir, 'metadata.csv'), encoding = 'utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            wav_path = os.path.join(fdir, 'wavs', '%s.wav' % parts[0])
            if hps.prep:
                f_list.append(get_mel_text_pair(parts[1], wav_path))
            else:
                f_list.append([parts[1], wav_path])
    if hps.prep and hps.pth is not None:
        with open(hps.pth, 'wb') as w:
            pickle.dump(f_list, w)
    return f_list


class csmscdataset(Dataset):
    def __init__(self):
        if hps.prep and hps.pth is not None and os.path.isfile(hps.pth):
            self.f_list = []
            with open(hps.pth,"r") as log:
                lines = log.readlines()
                for line in lines:
                    wav_path, sent = line.strip().split("|")
                    self.f_list.append((wav_path,sent))
            print(len(self.f_list))

        self.lexicon = self.get_lexicon(hps.lexicon_path)
    def get_lexicon(self, lexicon_path):
        lexicon = {}
        with open(lexicon_path) as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
        return lexicon

    def __getitem__(self, index):
        text, mel = self.f_list[index] if hps.prep \
                    else self.get_mel_text_pair(self.f_list[index])


    def __len__(self):
        return len(self.f_list)

    def get_mel_text_pair(self,params):
        wav_path, text = params
        text = text.strip()
        text = self.get_text(text)
        mel = self.get_mel(wav_path)
        return (text, mel)

    def get_text(self, text):
        text_map = []
        for piny in text.split():
            if piny in self.lexicon:
                phones = self.lexicon[piny]
                text_map = text_map + phones
            if piny[0] == "#":
                text_map = text_map + [piny]
        text = " ".join(text_map)
        text = "{" + text + "}"
        print(text)
        print(len(text.split()))
        return torch.IntTensor(text_to_sequence(text, []))

    def get_mel(self, wav_path):
        wav = load_wav(wav_path)
        return torch.Tensor(melspectrogram(wav).astype(np.float32))


class csmsccollate():
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths


if __name__ == "__main__":
    csmsc_obj = csmscdataset()
    out = csmsc_obj.get_text("wo3 neng2 #1 hen3 xin1 #1 da3 duan4 #1 ni3 bu5 #4")
    print(out)
    print(out.shape)
