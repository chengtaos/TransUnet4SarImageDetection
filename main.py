import sys
import torch
import argparse
from utils.train import TrainTestPipe
from utils.inference import Inference


def main_pipeline(parser):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if parser.mode == 'train':
        ttp = TrainTestPipe(train_path=parser.train_path,
                            test_path=parser.test_path,
                            model_path=parser.model_path,
                            device=device)
        ttp.train()

    elif parser.mode == 'inference':
        inf = Inference(model_path=parser.model_path,
                           device=device)

        _ = inf.infer(parser.image_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'])
    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument('--train_path', required='train' in sys.argv,  type=str, default=None)
    parser.add_argument('--test_path', required='train' in sys.argv, type=str, default=None)

    parser.add_argument('--image_path', required='infer' in sys.argv, type=str, default=None)
    parser = parser.parse_args()

    main_pipeline(parser)
