
import argparse
import os

from train import train
from test import test
from eval import eval
from model import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest = 'mode')

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--csv_path", help="Set in-image_path")
    train_parser.add_argument("--data_path", help="Set in-image path")
    train_parser.add_argument('--epochs', type=int, default=200)
    train_parser.add_argument('--batch_size', type=int, default=128, help='total batch size for all GPUs')
    train_parser.add_argument("--model_dir", help="Set out model path", default="./weights/")
    train_parser.add_argument("--gpu", help="Set gpu number", default="0", dest="gpu")

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--data_dir", help="Set in-data path")
    test_parser.add_argument("--model_path", help="Set trained model path")
    test_parser.add_argument("--class_csv_path", help="Set iclass csv path")
    test_parser.add_argument("--gpu", help="Set gpu number", default="0", dest="gpu")

    val_parser = subparsers.add_parser("eval")
    val_parser.add_argument("--model_path", help="Set trained model path")
    val_parser.add_argument("--class_csv_path", help="Set iclass csv path")
    val_parser.add_argument("--csv_path", help="Set in-image_path")
    val_parser.add_argument("--data_path", help="Set in-image path")
    val_parser.add_argument('--batch_size', type=int, default=128, help='total batch size for all GPUs')
    val_parser.add_argument("--gpu", help="Set gpu number", default="0", dest="gpu")

    parser_args = parser.parse_args()


    # os.environ["CUDA_VISIBLE_DEVICES"] = parser_args.gpu

    if parser_args.mode == 'train':
        train(parser_args)
    elif parser_args.mode == 'test':
        test(parser_args)
    elif parser_args.mode == 'eval':
        eval(parser_args)
        pass
