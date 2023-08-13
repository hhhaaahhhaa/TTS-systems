import torch


DEBUG = False
CUDA_LAUNCH_BLOCKING = False
MAX_WORKERS = 4
DATAPARSERS = {}
ALLSTATS = {}
LOGGER = "tb"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
