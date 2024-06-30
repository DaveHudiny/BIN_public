# Author = David Hudak
# Login = xhudak03
# Subject = BIN
# Year = 2022/2023
# Short Description = network_loader.py file of project to subject Biology Inspired Computers

import torch
import torchvision
from torchvision import datasets, transforms
import os
import lenet

transform_alex = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_lenet = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


class Network_Loader():
    """Class for loading, storing, training and testing neural networks.
    """

    def __init__(self, model_name: str = "alexnet", model_path: str = "./nns/alexnet.pkl"):
        """Init function of Network_Loader class.

        Args:
            model_name (str, optional): Contains name of loaded network. Eg. alexnet, lenet. Defaults to "alexnet".
            model_path (str, optional): Contains path for loading of neural network. If it does not lead to existing network, 
                                        it creates new network. Defaults to "./nns/alexnet.pkl".
        """
        self.model_name = model_name
        self.model_path = model_path
        if model_name == "alexnet":
            self.transform = transform_alex
        elif model_name == "lenet":
            self.transform = transform_lenet
        self.trainset = datasets.MNIST(
            root='./data', train=True, download=True, transform=self.transform)
        self.testset = datasets.MNIST(
            root='./data', train=False, download=True, transform=self.transform)

        self.reinit_model()

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=32, shuffle=False)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=32, shuffle=False)
        self.device = torch.device("cuda")

    def reinit_model(self):
        """Function reinitializes network model, or load it from stored path. Ideal for multiple runs of algorithm.
        """
        self.model = Network_Loader.load_model(self.model_path)
        if self.model is None:
            if self.model_name == "alexnet":
                self.model = torchvision.models.alexnet(
                    weights=torchvision.models.AlexNet_Weights.DEFAULT)
                self.model.features[0] = torch.nn.Conv2d(
                    1, 64, kernel_size=11, stride=4, padding=2)
                self.model.classifier[6] = torch.nn.Linear(4096, 10)
            elif self.model_name == "lenet":
                self.model = lenet.LeNet()
            print(f"[INFO] New model of {self.model_name} was created.")
        else:
            print(
                f"[INFO] Model of {self.model_name} was loaded from {self.model_path}.")

    def train(self, epochs: int = 5):
        """Train of neural network on train dataset.

        Args:
            epochs (int, optional): Epochs of training. Defaults to 5.
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        lossFn = torch.nn.CrossEntropyLoss()
        self.model.cuda()

        print("[INFO] The training has begun")
        for epoch in range(epochs):
            totalTrainLoss = 0
            trainCorrect = 0
            for (x, y) in self.trainloader:
                # x = x.permute(0, 3, 1, 2)

                # send the input to the device
                (x, y) = (x.to(self.device), y.to(self.device))
                # perform a forward pass and calculate the training loss
                y = y.long()
                pred = self.model(x)
                loss = lossFn(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss += loss
                trainCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()
            print("Epoch {}, Loss: {}, Correct outputs: {}".format(
                epoch + 1, totalTrainLoss, trainCorrect))

    def test_model(self):
        """Tests model on MNIST test dataset (or different loaded dataset).

        Returns:
            float: Accuracy of classification (in %).
        """
        self.model.eval()  # switch to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # no need to compute gradients during testing
            for data in self.testloader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print("Model accuracy: %.2f (%d of %d)" % (accuracy, correct, total))
        return accuracy

    def save_model(self, path):
        """Saves model.

        Args:
            path (str): Where to save model (name of file included).
        """
        with open(path, "wb") as handle:
            torch.save(self.model, handle)

    def load_model(path):
        """Loads network model.

        Args:
            path (str): Where is network stored.

        Returns:
            torch.nn.Module: Returns model (if exists, either None).  
        """
        if os.path.exists(path):
            with open(path, "rb") as handle:
                model = torch.load(handle)
            return model
        else:
            return None


if __name__ == "__main__":
    network = Network_Loader(model_name="lenet", model_path="./nns/lenet.pkl")
    network.train(epochs=3)
    network.test_model()
    network.save_model("./nns/lenet.pkl")
