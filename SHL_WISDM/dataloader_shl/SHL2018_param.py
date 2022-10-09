from pathlib import Path
import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
# os.path.abspath(os.path.dirname(os.getcwd()))
# data_path = os.path.abspath(os.path.join(os.getcwd(), "../../Data/SHL2018"))
data_path = Path(r"./Data/SHL2018")
print(data_path)

classes_names = ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"]

fs = 100  # sampling frequency
duration_window = 5     # parameters for spectrogram creation
duration_overlap = 4.9
duration_segment = 60

spectro_batch_size = 1000  # parameter for spectrogram computation.
len_train = 13000
