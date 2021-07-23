import random
import torch
import torchaudio
from torchaudio import transforms

from torch.utils.data import Dataset




class AudioUtil:
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if sig.shape[0] == new_channel:
            return aud

        if new_channel == 1:
            resig = sig[:1, :]
        else:
            resig = torch.cat([sig, sig])

        return (resig, sr)

    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if sr == newsr:
            return aud

        num_channels = sig.shape[0]
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])

        if num_channels > 1:
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return (resig, newsr)

    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if sig_len > max_len:
            sig = sig[:, :max_len]

        elif sig_len < max_len:
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)

    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        spec = transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels
        )(sig)

        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)

        return spec

    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels

        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(
                aug_spec, mask_value
            )

        time_mask_param = max_mask_pct * n_steps

        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec


class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        LABELS = list(self.df.label.unique())
        # print(LABELS)

        label_idx = {label: i for i, label in enumerate(LABELS)}

        audio_file = self.data_path + self.df.loc[idx, "fname"]
        # class_id = one_hot_y[label_idx[self.df.loc[idx, "label"]]]
        class_id = label_idx[self.df.loc[idx, "label"]]

        sig, sr = AudioUtil.open(audio_file)
        sig = (sig - sig.mean()) / sig.std()

        aud = (sig, sr)

        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)
        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)

        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(
            sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
        )

        return aug_sgram, class_id


def inference(model, val_dl, device):
    correct_prediction = 0
    total_prediction = 0

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0].to(device), data[1].to(device)

            # inputs_m, inputs_s = inputs.mean(), inputs.std()

            # inputs_m, inputs_s = -28, 21
            # inputs = (inputs - inputs_m) / inputs_s

            outputs = model(inputs)
            # input(outputs[0])

            _, prediction = torch.max(outputs, 1)
            # correct_prediction += (prediction == torch.max(labels)).sum().item()
            # print(prediction, labels)

            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f"  Validation Accuracy: {acc:.2f}")


def read_text_file_by_line(path):
    with open(path, mode="r", encoding="utf-8") as f:
        for i in f:
            i = i.strip()
            if i:
                yield i


def parse_categories_file(csv_path):
    result = {}
    for line in read_text_file_by_line(csv_path):
        split = line.split(",")
        if len(split) < 2 or any(not i for i in split):
            continue
        result[split[0]] = int(split[1])
    return result
