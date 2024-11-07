from ImageUtils import parse_record
from DataReader import load_data, train_vaild_split
from Model import Cifar

import os
import argparse

def configure():
    parser = argparse.ArgumentParser()
    ### YOUR CODE HERE
    parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
    parser.add_argument("--num_classes", type=int, default=10, help='number of classes')
    parser.add_argument("--save_interval", type=int, default=10, 
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--weight_decay", type=float, default=2e-4, help='weight decay rate')
    parser.add_argument("--modeldir", type=str, default='model_v1', help='model directory')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gpu", type=int, default=0)
    ### YOUR CODE HERE
    # return parser.parse_args()
    args, unknown = parser.parse_known_args()
    print("Unrecognized arguments:", unknown)
    return args

def main(config):
    print("--- Preparing Data ---")

    ### YOUR CODE HERE
    data_dir = ""
    ### YOUR CODE HERE

    x_train, y_train, x_test, y_test = load_data(data_dir)
    x_train_new, y_train_new, x_valid, y_valid = train_vaild_split(x_train, y_train)

    model = Cifar(config).cuda()

    ### YOUR CODE HERE
    model.train(x_train_new, y_train_new, 100)
    model.test_or_validate(x_valid, y_valid, [80, 90, 100])
    model.test_or_validate(x_test, y_test, [100])

    ### END CODE HERE

if __name__ == "__main__":
    config = configure()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    main(config)