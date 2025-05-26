from huggingface_hub import hf_hub_download
import torch
import pickle
import shutil
import os

def load_model_and_encoder():

    # Define filenames
    model_filename = "model_shakespeare_new_v5_latest.pth"
    encoder_filename = "encoder_shakespeare_v5.pkl"

    model_path = hf_hub_download("khotveer1/custom-gpt-pytorch-shakespeare", model_filename)
    encoder_path = hf_hub_download("khotveer1/custom-gpt-pytorch-shakespeare", encoder_filename)

    # Target destination
    target_model_path = os.path.join(".", model_filename)
    target_encoder_path = os.path.join(".", encoder_filename)

    # print("model download: ", model_path)
    # print("encoder download: ", encoder_path)

    # Move files to current directory
    shutil.copy(model_path, target_model_path)
    shutil.copy(encoder_path, target_encoder_path)

    print(f"Model saved to: {target_model_path}")
    print(f"Encoder saved to: {target_encoder_path}")


if __name__ == "__main__":
    _ = load_model_and_encoder()





