import torch
import torch.nn.functional as F

from ..model_training.cross_entropy_pre_training.cross_entropy_model import FBankCrossEntropyNet


def get_cosine_distance(a, b):
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    return (1 - F.cosine_similarity(a, b)).numpy()


MODEL_PATH = 'weights/triplet_loss_trained_model.pth'
model_instance = FBankCrossEntropyNet()
model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
model_instance = model_instance.double()
model_instance.eval()


def get_embeddings(x):
    x = torch.from_numpy(x)
    with torch.no_grad():
        embeddings = model_instance(x)
    return embeddings.numpy()
