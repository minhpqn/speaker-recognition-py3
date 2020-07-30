"""
Command-line interface for speaker recognition system
"""
import os
import argparse
from logzero import logger
import tempfile
import uuid
import speech_recognition as sr

from utils import read_wav
from interface import ModelInterface


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite_model",
        action="store_true",
        help="Whether to overwrite the model"
    )
    parser.add_argument(
        "--num_samples", type=int, default=3, help="Number of sound samples"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to model file")
    parser.add_argument(
        "--task",
        required=True,
        choices=["enroll", "predict"],
        help='Task to do. Either "enroll" or "predict"')
    args = parser.parse_args()
    
    if args.task == "predict" and not os.path.isfile(args.model_path):
        raise ValueError("Please provide valid model path")

    r = sr.Recognizer()
    r.pause_threshold = 1.0
    if args.task == "enroll":
        if os.path.isfile(args.model_path) and not args.overwrite_model:
            model = ModelInterface.load(args.model_path)
        else:
            model = ModelInterface()
        print("***** Enroll sound data for one speaker *****")
        name = input("Enter your name: ")
        name = name.strip()
        print(f"Hello {name}. Please input your voice {args.num_samples} times")
        with tempfile.TemporaryDirectory() as tempdir:
            i = 1
            while i <= args.num_samples:
                with sr.Microphone() as source:
                    audio = r.listen(source)
                # Generate random filename
                filename = os.path.join(tempdir, name + "_" + str(uuid.uuid1()) + ".wav")
                with open(filename, "wb") as file:
                    file.write(audio.get_wav_data(convert_rate=16000))
                # enroll a file
                fs, signal = read_wav(filename)
                model.enroll(name, fs, signal)
                logger.info("wav file %s has been enrolled" % (filename))
                i += 1
            
        model.train()
        model.dump(args.model_path)
    else:
        model = ModelInterface.load(args.model_path)
        print("Please input your voice: ")
        with tempfile.TemporaryDirectory() as tempdir:
            with sr.Microphone() as source:
                audio = r.listen(source)
            filename = os.path.join(tempdir, str(uuid.uuid1()) + ".wav")
            with open(filename, "wb") as file:
                file.write(audio.get_wav_data(convert_rate=16000))
            fs, signal = read_wav(filename)
            pred, _ = model.predict(fs, signal)
            print(f"Your name is {pred}!")