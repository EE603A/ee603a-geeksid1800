import torch
from torch import nn
from torch.utils.data import DataLoader

from melspecdataset import MelDataset
from cnn import CNNNetwork
from densenn import DenseNet

BATCH_SIZE = 32
EPOCHS = 18
LEARNING_RATE = 0.001

MEL_DIR = 'C:/Users/geeks/OneDrive - IIT Kanpur/EE603/Audio_Classification-MLSP/Audio_Classification-MLSP/train'
ANNOTATIONS_FILE = 'C:/Users/geeks/OneDrive - IIT Kanpur/EE603/Audio_Classification-MLSP/Audio_Classification-MLSP/annotations.csv'
IDEAL_FRAMES = 500 #Pad if less, trim if more than ideal

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input = input.to(device) 
        target = target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # instantiating our dataset object and create data loader

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
    
    train_dataloader = create_data_loader(data, BATCH_SIZE)

    # construct model and assign it to device
    #cnn = CNNNetwork().to(device)
    #print(cnn)

    ann = DenseNet().to(device)
    print(ann)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    #optimiser = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)
    optimiser = torch.optim.Adam(ann.parameters(), lr=LEARNING_RATE)

    # train model
    #train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)
    train(ann, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    #torch.save(cnn.state_dict(), "cnnnet.pth")
    torch.save(ann.state_dict(), "annnet.pth")

    #print("Trained CNN net saved at cnnnet.pth")
    print("Trained ANN net saved at annnet.pth")