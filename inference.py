#THIS FILE IS USED FOR CHECKING THE ACCURACY ON THE TRAINING SET
#To predict for the test set, use predictions.py

import torch
from cnn import CNNNetwork
from densenn import DenseNet
from melspecdataset import MelDataset
from train import MEL_DIR, ANNOTATIONS_FILE, IDEAL_FRAMES
import random

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


def evaluate_train(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        conf = torch.softmax(predictions[0], dim=0) #the probability of it being each class
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected, conf[predicted_index]

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
    print(f"Using {device}")

    # load mel spectrogram dataset

    data = MelDataset(ANNOTATIONS_FILE,
                            MEL_DIR,
                            IDEAL_FRAMES,
                            device,
                            class_to_int)


    # get a random sample from the urban sound dataset for inference
    sample_ix = random.randint(0,999)
    input, target = data[sample_ix][0], data[sample_ix][1] # [batch size, num_channels, fr, time]
    print(input.shape)
    input.unsqueeze_(0) #to make it a 4d tensor, which our CNN requires
    print(input.shape)
    predicted, expected, conf = evaluate_train(cnn, input, target,
                                  int_to_class)
    print(f"Predicted: '{predicted}' - {conf}, expected: '{expected}'")

    # correct = 0
    # for ix in range(1000):
    #     input, target = data[ix][0], data[ix][1] # [batch size, num_channels, fr, time]
    #     input.unsqueeze_(0) #to make it a 4d tensor, which our CNN requires

    # # make an inference
    #     predictedCNN, expected, confCNN = evaluate_train(cnn, input, target, int_to_class)
    #     predictedANN, expected, confANN = evaluate_train(ann, input, target, int_to_class)

    #     if(predictedANN == expected): correct = correct + 1
    #     else: print(f"Index {ix} is wrong, Predicted: '{predictedANN}' - {confANN} ANN, expected: '{expected}'")
    # print(f"Accuracy: {correct/1000}")