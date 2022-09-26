import torch


DEBUG = False
CUDA_LAUNCH_BLOCKING = True
MAX_WORKERS = 2
DATAPARSERS = {}
ALLSTATS = {}
LOGGER = "tb"  # "tb", "comet", ""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
