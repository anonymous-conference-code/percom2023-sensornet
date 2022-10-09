
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


fs = 20
duration_window = 2   # parameters for spectrogram creation
duration_overlap = 1.5
duration_segment = 10


