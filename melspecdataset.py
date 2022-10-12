import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


class MelDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 mel_dir,
                 ideal_frames,
                 device,
                 mapping):
        
        self.annotations = pd.read_csv(annotations_file)
        self.mel_dir = mel_dir
        self.ideal_frames = ideal_frames
        self.device = device
        self.mapping = mapping

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        mel_spec_path = self._get_mel_spec_path(index)
        label = self._get_mel_spec_label(index)
        label = self.mapping[label]
        signal = np.load(mel_spec_path)
        #print(f"Size {signal.shape} after numpyload")
        signal = torch.from_numpy(signal)
        #print(f"Size {signal.shape} after fromNumpy")
        signal = signal.to(self.device)
        signal = self._mix_down_if_necessary(signal)
        #print(f"Size {signal.shape} after mixdown")
        signal = self._cut_if_necessary(signal)
        #print(f"Size {signal.shape} after cut")
        signal = self._right_pad_if_necessary(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        length_spec = signal.shape[2]
        if length_spec > self.ideal_frames:
            signal = signal[:,:, :self.ideal_frames]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[2]
        if length_signal < self.ideal_frames:
            num_missing_samples = self.ideal_frames - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_mel_spec_path(self, index):
        path = os.path.join(self.mel_dir, self.annotations.iloc[index, 1])
        return path

    def _get_mel_spec_label(self, index):
        return self.annotations.iloc[index, 2]
    


if __name__ == "__main__":
    MEL_DIR = 'C:/Users/geeks/OneDrive - IIT Kanpur/EE603/Audio_Classification-MLSP/Audio_Classification-MLSP/train'
    ANNOTATIONS_FILE = 'C:/Users/geeks/OneDrive - IIT Kanpur/EE603/Audio_Classification-MLSP/Audio_Classification-MLSP/annotations.csv'
    IDEAL_FRAMES = 500 #Pad if less, trim if more than ideal
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    #print(f"Using device {device}")

    class_to_int = {
            "Bark" : 0,
            "Meow" : 1,
            "Siren" : 2,
            "Shatter" : 3,
            "Knock" : 4,
            "Crying_and_sobbing" : 5,
            "Microwave_oven" : 6,
            "Vehicle_horn_and_car_horn_and_honking" : 7,
            "Doorbell" : 8,
            "Walk_and_footsteps" : 9
        }
    int_to_class = [
            "Bark",
            "Meow",
            "Siren",
            "Shatter",
            "Knock",
            "Crying_and_sobbing",
            "Microwave_oven",
            "Vehicle_horn_and_car_horn_and_honking",
            "Doorbell",
            "Walk_and_footsteps"
        ]

    data = MelDataset(ANNOTATIONS_FILE,
                            MEL_DIR,
                            IDEAL_FRAMES,
                            device,
                            class_to_int)

    print(f"There are {len(data)} samples in the dataset.")
    signal0, label0 = data[0]
    print(signal0.shape,int_to_class[label0])