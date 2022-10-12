import os, csv
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from cnn import CNNNetwork
from densenn import DenseNet
import random

class TestDataset(Dataset):

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
        return signal

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
        path = os.path.join(self.mel_dir, self.annotations.iloc[index, 0])
        return path


def predict(model, input, class_mapping): #Used to Predict during testing
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        conf = torch.softmax(predictions[0], dim=0) #the probability of it being each class
        predicted = class_mapping[predicted_index]
    return predicted, conf[predicted_index]


if __name__ == "__main__":
    MEL_DIR = 'C:/Users/geeks/OneDrive - IIT Kanpur/EE603/Audio_Classification-MLSP-test'
    IDEAL_FRAMES = 500 #Pad if less, trim if more than ideal

    # lst = os.listdir(MEL_DIR)
    # df = pd.DataFrame(lst)
    # df.to_csv('C:/Users/geeks/OneDrive - IIT Kanpur/EE603/filenames.csv', index=False)

    ANNOTATIONS_FILE = 'C:/Users/geeks/OneDrive - IIT Kanpur/EE603/filenames.csv'

    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("cnnnet.pth")
    cnn.load_state_dict(state_dict)

    # load back the model
    ann = DenseNet()
    state_dict = torch.load("annnet.pth")
    ann.load_state_dict(state_dict)

    
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

    data = TestDataset(ANNOTATIONS_FILE,
                            MEL_DIR,
                            IDEAL_FRAMES,
                            device,
                            class_to_int)

    # print(f"There are {len(data)} samples in the dataset.")
    # signal0 = data[0]
    # print(f"signal0.shape {signal0.shape}")

    preds = []
    for ix in range(201):
        input =  data[ix] # [batch size, num_channels, fr, time]
        input.unsqueeze_(0) #to make it a 4d tensor, which our CNN requires
        
    # make an inference
        predictedCNN, confCNN = predict(cnn, input, int_to_class)
        predictedANN, confANN = predict(ann, input, int_to_class)

        if(predictedANN == predictedCNN): finalpred = predictedCNN
        else:
            if(confCNN > confANN): finalpred = predictedCNN
            else: finalpred = predictedANN
        preds.append(finalpred)
        
    output = pd.read_csv(ANNOTATIONS_FILE)
    output.insert(1,"prediction",preds,allow_duplicates=True)
    output.to_csv('C:/Users/geeks/OneDrive - IIT Kanpur/EE603/predictions.csv', index=False)
    

    # # get a random sample from the urban sound dataset for inference
    # sample_ix = random.randint(0,201)
    # input = data[sample_ix] # [batch size, num_channels, fr, time]
    # print(f"before {input.shape}")
    # input.unsqueeze_(0) #to make it a 4d tensor, which our CNN requires
    # print(input.shape)
    # predicted, conf = predict(cnn, input, int_to_class)
    # print(f"Predicted: {sample_ix} '{predicted}' - {conf}")
