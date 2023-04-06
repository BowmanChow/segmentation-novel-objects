
LOG_DIR = "./logs"
METHOD_NAME = "base"

MAX_EPOCH = 150
TRAIN_BATCH_PER_GPU = 18
VALID_BATCH_PER_GPU = 50
GPU_S = "5"
LEARNING_RATE = 5e-4
BASE_MODEL_PATH = "./logs/2023-04-06-11:22:58_base"
# BASE_MODEL_PATH = None

IMAGE_FOLDER = '../stablediffusion/outputs/img_2_img_heatmap'
IMAGE_IDS = list(range(460, 1116))
