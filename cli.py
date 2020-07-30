"""
Command-line interface for speaker recognition system
"""
import os
import argparse
from logzero import logger
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
        "--audio_data_dir",
        default="./tmp",
        help="Directoty to save audio data"
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
    
    os.makedirs(args.audio_data_dir, exist_ok=True)
    if args.task == "predict" and not os.path.isfile(args.model_path):
        raise ValueError("Please provide valid model path")

    r = sr.Recognizer()
    if args.task == "enroll":
        if os.path.isfile(args.model_path) and not args.overwrite_model:
            model = ModelInterface.load(args.model_path)
        else:
            model = ModelInterface()
        print("***** Enroll sound data for one speaker *****")
        name = input("Enter your name: ")
        name = name.strip()
        print(f"Hello {name}. Please input your voice 3 times")
        i = 1
        while i <= 3:
            with sr.Microphone() as source:
                audio = r.listen(source)
            # Generate random filename
            filename = os.path.join(args.audio_data_dir, name + "_" + str(uuid.uuid1()) + ".wav")
            with open(filename, "wb") as file:
                file.write(audio.get_wav_data())
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
        with sr.Microphone() as source:
            audio = r.listen(source)
        filename = os.path.join(str(uuid.uuid1()) + ".wav")
        with open(filename, "wb") as file:
            file.write(audio.get_wav_data())
        fs, signal = read_wav(filename)
        pred, _ = model.predict(fs, signal)
        print(f"Your name is {pred}!")
    