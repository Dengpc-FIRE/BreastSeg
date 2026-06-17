import torch
import time
import os
import warnings
import argparse

from trainer import Trainer

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, help='0,1,2,3 or 4')
    parser.add_argument('--data_root', default='/home/ubuntu/zhangyongtao/MTLN/Gastric_data/preprocessed_data',
                        type=str, help='root directory path of data')
    parser.add_argument("--test_best", required=False, default=False, help="select the best training weights to test",
                        action="store_true")
    parser.add_argument("--input_channels", type=int, default=17,
                        help="number of image input channels before the segmentation label channel")
    args = parser.parse_args()
    out_path = "/home/ubuntu/zhangyongtao/MA-MTLN"
    data_root = args.data_root
    fold = args.fold
    test_best = args.test_best
    input_channels = args.input_channels
    out_checkpoints = os.path.join(out_path, "Fold" + str(fold) + "_checkpoints")
    if not os.path.exists(str(out_checkpoints)):
        os.mkdir(str(out_checkpoints))

    model_trainer = Trainer(fold, data_root, out_path, out_checkpoints, input_channels=input_channels)

    if not test_best:
        model_trainer.run_trainer()
    #
    # model_trainer.initialize(not test_best)
    #
    # if test_best:
    #     model_trainer.load_checkpoint(train=False)
    #     model_trainer.validate(validation_restore_path="validation")
