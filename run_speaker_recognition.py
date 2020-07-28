"""
Training the speaker identification model
"""
import re
import os
import argparse
from logzero import logger

from interface import ModelInterface
from utils import read_wav


def train(train_data_dir, model_path):
    m = ModelInterface()
    files = [f for f in os.listdir(train_data_dir) if re.search(r"\.wav", f)]
    for f in files:
        label, _ = f.split("_")
        file = os.path.join(train_data_dir, f)
        try:
            fs, signal = read_wav(file)
            m.enroll(label, fs, signal)
            logger.info("wav %s has been enrolled" % (file))
        except Exception as e:
            logger.info(file + " error %s" % (e))

    m.train()
    m.dump(model_path)


def evaluate(eval_data_dir, model_path):
    m = ModelInterface.load(model_path)
    files = [f for f in os.listdir(eval_data_dir) if re.search(r"\.wav", f)]
    total, n_correct = 0, 0
    for f in files:
        total += 1
        label, _ = f.split("_")
        file = os.path.join(eval_data_dir, f)
        fs, signal = read_wav(file)
        pred, _ = m.predict(fs, signal)
        logger.info("Input: {}, Output: {}".format(file, pred))
        if label == pred:
            n_correct += 1
    logger.info("Accuracy: {}".format(n_correct/total))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to model file"
    )
    parser.add_argument(
        "--train_data_dir",
        default="/Users/minhpham/nlp/data/speech/elsdsr/train",
        help="Path to ELSDSR training data directory"
    )
    parser.add_argument(
        "--eval_data_dir",
        default="/Users/minhpham/nlp/data/speech/elsdsr/test",
        help="Path to evaluation data"
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to train the model")
    parser.add_argument("--do_eval", action="store_true", help="Whether to evaluate the model")
    args = parser.parse_args()
    
    logger.info(args)
    
    if args.do_train:
        logger.info("Training speaker recognition model")
        train(args.train_data_dir, args.model_path)
    
    if args.do_eval:
        evaluate(args.eval_data_dir, args.model_path)
    

if __name__ == "__main__":
    main()

