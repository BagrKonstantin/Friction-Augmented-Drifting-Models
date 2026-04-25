import os
import sys
import numpy as np
import torch
from OFM.ofmsrc.alae_distributions import TensorSampler

def download_data(data_path="../OFM/data"):
    import gdown
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    urls = {
        os.path.join(data_path, "age.npy"): "https://drive.google.com/uc?id=1Vi6NzxCsS23GBNq48E-97Z9UuIuNaxPJ",
        os.path.join(data_path, "gender.npy"): "https://drive.google.com/uc?id=1SEdsmQGL3mOok1CPTBEfc_O1750fGRtf",
        os.path.join(data_path, "latents.npy"): "https://drive.google.com/uc?id=1ENhiTRsHtSjIjoRu1xYprcpNd8M9aVu8",
        os.path.join(data_path, "test_images.npy"): "https://drive.google.com/uc?id=1SjBWWlPjq-dxX4kxzW-Zn3iUR3po8Z0i",
    }

    for name, url in urls.items():
        if not os.path.exists(name):
            print(f"Downloading {name}...")
            gdown.download(url, name, quiet=False)

def download_alae_model(model_dir="../OFM/ALAE/training_artifacts/ffhq/"):
    import torch.hub
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model_157.pth")
    last_checkpoint_path = os.path.join(model_dir, "last_checkpoint")

    if not os.path.exists(model_path):
        url = "https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth"
        print(f"Downloading ALAE model weights to {model_path}...")
        torch.hub.download_url_to_file(url, model_path)

    if not os.path.exists(last_checkpoint_path):
        with open(last_checkpoint_path, "w") as f:
            # Note: The original files use a relative path inside the file
            f.write("training_artifacts/ffhq/model_157.pth")

def get_indices(data_type, age_arr, gen_arr, size):
    if data_type == "MAN":
        return np.arange(size)[(gen_arr == "male").reshape(-1)]
    elif data_type == "WOMAN":
        return np.arange(size)[(gen_arr == "female").reshape(-1)]
    elif data_type == "ADULT":
        return np.arange(size)[(age_arr >= 18).reshape(-1) * (age_arr != -1).reshape(-1)]
    elif data_type == "CHILDREN":
        return np.arange(size)[(age_arr < 18).reshape(-1) * (age_arr != -1).reshape(-1)]
    return np.arange(size)

def load_alae_data(data_path, input_data_type, target_data_type, device, train_size=60000, test_size=10000):
    try:
        latents = np.load(os.path.join(data_path, "latents.npy"))
        gender = np.load(os.path.join(data_path, "gender.npy"))
        age = np.load(os.path.join(data_path, "age.npy"))
        test_inp_images = np.load(os.path.join(data_path, "test_images.npy"))

        train_latents, test_latents = latents[:train_size], latents[train_size:]
        train_gender, test_gender = gender[:train_size], gender[train_size:]
        train_age, test_age = age[:train_size], age[train_size:]

        x_inds_train = get_indices(input_data_type, train_age, train_gender, train_size)
        x_inds_test = get_indices(input_data_type, test_age, test_gender, test_size)
        y_inds_train = get_indices(target_data_type, train_age, train_gender, train_size)
        y_inds_test = get_indices(target_data_type, test_age, test_gender, test_size)

        X_train = torch.tensor(train_latents[x_inds_train])
        Y_train = torch.tensor(train_latents[y_inds_train])

        X_test = torch.tensor(test_latents[x_inds_test])
        Y_test = torch.tensor(test_latents[y_inds_test])

        X_sampler = TensorSampler(X_train, device=device)
        Y_sampler = TensorSampler(Y_train, device=device)
        
        return {
            "X_train": X_train, "Y_train": Y_train,
            "X_test": X_test, "Y_test": Y_test,
            "X_sampler": X_sampler, "Y_sampler": Y_sampler,
            "x_inds_test": x_inds_test,
            "test_inp_images": test_inp_images
        }
    except FileNotFoundError:
        print(f"Error: Required .npy files not found in {data_path}.")
        sys.exit(1)
