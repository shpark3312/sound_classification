import os
import numpy as np 
from utils import *


if __name__ == '__main__':

    data_dir = 'dataset/210811_data'
    audiofile_names = [f for f in os.listdir(data_dir) if f[-4:] == ".wav"]
    print(audiofile_names)

    for audiofile in audiofile_names:

        sig, sr = AudioUtil.open(os.path.join(data_dir, audiofile))

        audio_length = np.floor(sig.shape[1]/sr)
        cut_audio_file_num = int(np.floor(audio_length/2))

        print('processing filename = ', audiofile)
        print(f'total_soundfile_length = {audio_length}')
        print(f'cut_audio_file_num = {cut_audio_file_num}')

        for i in range(cut_audio_file_num):
            cut_sig = sig[:, sr * i: sr*(i+1)]
            torchaudio.save(os.path.join(data_dir, 'cut', f'{audiofile.split(".")[0]}_{i}.wav'), cut_sig, sr)