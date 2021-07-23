import numpy as np
import torch
import time
from utils import *
import os
from model import *

def test(parser_args):
    cat_table = parse_categories_file(parser_args.class_csv_path)
    reverse_cat_table = {v: k for k, v in cat_table.items()}  # indices -> name


    model = torch.load(parser_args.model_path)
    model.eval()
    device = torch.device(f"cuda:{parser_args.gpu}" if torch.cuda.is_available() else "cpu")
    print("training with decive :", device)

    model = model.to(device)

    audiofile_names = [f for f in os.listdir(parser_args.data_dir) if f[-4:] == ".wav"]
    # print(audiofile_names)

    for audiofile in audiofile_names:

        sig, sr = AudioUtil.open(os.path.join(parser_args.data_dir, audiofile))

        sig = (sig - sig.mean()) / sig.std()
        aud = (sig, sr)

        duration = 4000
        sr = 44100
        channel = 2


        reaud = AudioUtil.resample(aud, sr)
        rechan = AudioUtil.rechannel(reaud, channel)
        dur_aud = AudioUtil.pad_trunc(rechan, duration)
        sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
        res = test_sample(model, sgram, device, reverse_cat_table)

        print(f'filename : {audiofile}, Model Prediction : {res}')

def test_sample(model, sgram, device, reverse_cat_table):

    # plt.figure()
    # plt.imshow(sgram[0, :, :].detach().numpy())
    # plt.show()

    with torch.no_grad():
        current_time = time.time()

        inputs = sgram[np.newaxis, ...].to(device)

        # inputs_m, inputs_s = -28, 21
        # inputs = (inputs - inputs_m) / inputs_s

        outputs = model(inputs)

        _, prediction = torch.max(outputs, 1)

        print(f"prediction time = {time.time() - current_time}")

    return reverse_cat_table[prediction.tolist()[0]]
