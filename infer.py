import sys
import torch.optim as optim
from arguments import parse_args
from models import Res_U_Net, U_Net
from dataset import data_loader, split_ids
from model_run import train_epoch, validate
import torch
import torch.nn as nn
import numpy as np
import config
import pathlib
import os
from matplotlib import pyplot as plt
from loss import KnowledgeDistillationLoss
from utils import Tee, copy_codes, get_format_time, load_model, make_log_dir, save_model, set_random_seed

args, _ = parse_args()

set_random_seed(123)
current_file_path = pathlib.Path(__file__)
log_dir = make_log_dir(args.log, config.METHOD_NAME)
copy_codes(log_dir)
sys.stdout = Tee(os.path.join(log_dir, "print.log"))
np.set_printoptions(precision=4)

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_S

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"""
using {DEVICE} | cuda num {torch.cuda.device_count()}
""")

N_CLASSES = 1

valid_image_ids, train_image_ids = split_ids(config.IMAGE_IDS, 50)
_, valid_loader = data_loader(
    folder='../stablediffusion/outputs/img_2_img_heatmap',
    train_image_ids=train_image_ids,
    valid_image_ids=valid_image_ids,
    train_batch_size=config.TRAIN_BATCH_PER_GPU * torch.cuda.device_count(),
    valid_batch_size=config.VALID_BATCH_PER_GPU * torch.cuda.device_count(),
    is_normalize=True)


model = Res_U_Net(3, N_CLASSES).to(DEVICE)
# model = nn.DataParallel(model)
if config.BASE_MODEL_PATH is not None:
    load_model(
        model=model,
        load_dir=config.BASE_MODEL_PATH,
        suffice="valid_best",
    )


optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
criterions = {
    # "X_Entropy": (nn.CrossEntropyLoss().to(DEVICE), 1.0),
    # "Distill": (KnowledgeDistillationLoss().to(DEVICE), 1.0),
    "MSE": (nn.MSELoss().to(DEVICE), 1.0),
}

mse = nn.MSELoss()


def save_predict(X, predict, y):
    for i, label in enumerate(y):
        plt.subplot(1, 3, 1)
        plt.imshow(np.transpose(X[i].squeeze().cpu(), [1, 2, 0]))
        plt.subplot(1, 3, 2)
        plt.imshow(predict[i].squeeze().cpu())
        plt.colorbar()
        plt.title("predict")
        plt.subplot(1, 3, 3)
        plt.imshow(label.squeeze().cpu())
        plt.colorbar()
        plt.title("truth")
        plt.savefig(os.path.join(log_dir, f"{i}.png"))
    return mse(predict, y)


metrics = {
    "MSE": lambda X, pred, argmax, y: save_predict(X, pred, y),
}

valid_loss, valid_metrics = validate(
    valid_loader, model, criterions, metrics, DEVICE, print_every=1)
metric_now = valid_metrics["MSE"].mean()
