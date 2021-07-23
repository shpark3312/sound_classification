
import datetime
import os
import time

import pandas as pd
import torch
from torch.utils.data import random_split
from torch import nn

from utils import *
from model import *


def train(parser_args):

    print("loading data...")
    df = pd.read_csv(parser_args.csv_path)

    myds = SoundDS(df, parser_args.data_path)

    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])
    print(f"total : {num_items}, train : {num_train}, val : {num_val}")

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=parser_args.batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=parser_args.batch_size, shuffle=False)

    print("data loading done!")

    myModel = AudioClassifier_test()
    device = torch.device(f"cuda:{parser_args.gpu}" if torch.cuda.is_available() else "cpu")
    print("training with decive :", device)
    # device = torch.device("cpu")
    myModel = myModel.to(device)
    next(myModel.parameters()).device

    training(myModel, train_dl, val_dl, parser_args.epochs, device)

    todays_date = datetime.now()

    torch.save(myModel, os.path.join(parser_args.model_dir, f'{todays_date.year}{todays_date.month:02}{todays_date.day:02}_{todays_date.hour:02}_{todays_date.minute:02}.pt'))
    # torch.save(myModel.state_dict(), "./20210525_model_noBN_normalize_each_file.pth")


def training(model, train_dl, val_dl, num_epochs, device):
    print('Start training ...')

    cur_time = time.time()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=int(len(train_dl)),
        epochs=num_epochs,
        anneal_strategy="linear",
    )

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device)

            # inputs_m, inputs_s = inputs.mean(), inputs.std()
            # inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()

            outputs = model(inputs)
            # loss = criterion(outputs, torch.max(labels, 1)[1])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, prediction = torch.max(outputs, 1)
            # correct_prediction += (prediction == torch.max(labels)).sum().item()
            correct_prediction += (prediction == labels).sum().item()

            # print(f"prediction = {prediction}")
            # print(f"labels = {labels}")

            total_prediction += prediction.shape[0]

            print(
                f"Epoch : {epoch + 1}/{num_epochs}, step : {i + 1} / {int(len(train_dl))},  loss: {running_loss / (i+1) : .5f} \r",
                end="",
            )

        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction

        print(
            f"\nEpoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f} ",
            end="",
        )

        inference(model, val_dl, device)

    print(f"Finished Training, Total training time = {time.time() - cur_time}")
