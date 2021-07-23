import pandas as pd
import os

from utils import *
from model import *

def eval(parser_args):
    cat_table = parse_categories_file(parser_args.class_csv_path)
    reverse_cat_table = {v: k for k, v in cat_table.items()}  # indices -> name

    print("loading data...")
    df = pd.read_csv(parser_args.csv_path)

    myds = SoundDS(df, parser_args.data_path)
    dl = torch.utils.data.DataLoader(myds, batch_size=parser_args.batch_size, shuffle=True)

    print("data loading done!")

    model = torch.load(parser_args.model_path)
    model.eval()
    device = torch.device(f"cuda:{parser_args.gpu}" if torch.cuda.is_available() else "cpu")
    print("training with decive :", device)

    model = model.to(device)

    inference(model, dl, device)
