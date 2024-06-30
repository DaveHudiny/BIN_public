# Author = David Hudakloader
# Login = xhudak03
# Subject = BIN
# Year = 2022/2023
# Short Description = main.py file of project to subject Biology Inspired Computers. Calls CGP for AlexNet neural network

import argparse
from network_loader import Network_Loader
from trainers import train_conv_layers, train_layers, train_multiple_channels


def add_arguments():
    """Creates argparse parser with arguments.

    Returns:
        argparse.ArgumentParser: New argument parser.
    """
    parser = argparse.ArgumentParser(description="Program for generating")
    parser.add_argument("--network", type=str,
                        choices=["lenet", "alexnet"], default="alexnet", help="Selection of network.")
    parser.add_argument("--back_memory", default=False, type=bool,
                        help="Turn on/off back memory -- one of inputs of CGP is last output.")
    parser.add_argument("--runs", default=15, type=int,
                        help="How many runs for each experiment.")
    parser.add_argument("--epochs_conv", default=60, type=int,
                        help="Epochs for each convolutional run.")
    parser.add_argument("--epochs_fc", default=100, type=int,
                        help="Epochs for each fully connected layers run.")
    parser.add_argument("--dataset", default="MNIST",
                        type=str, help="Not implemented yet.")
    parser.add_argument("--basic", default=False, type=bool,
                        help="Use only basic functions.")
    parser.add_argument("--layers", default="both",
                        choices=["conv", "fc", "both"], help="Which layers should be regressed.")
    parser.add_argument("--mem_divisor", default=10, type=int,
                        help="Which division of memory use (1/arg) * real_size")
    return parser


def call_alexnet(args):
    """Calls learning of weights by CGP for AlexNet model.

    Args:
        args (Namespace): Args from argparse.
    """
    accuracies = []
    accuracies_continuous = []
    network = Network_Loader()
    for i in range(args.runs):
        epochs_conv = [args.epochs_conv + 200, args.epochs_conv,
                       args.epochs_conv, args.epochs_conv, args.epochs_conv]
        epochs_fc = [args.epochs_fc]
        sub_accuracies_continuous = []
        if args.layers in ["conv", "both"]:
            sub_accuracies_continuous.extend(train_conv_layers(network, epochs=epochs_conv, layers=[0, 3, 6, 8, 10],
                                                               path="./nns/alexnet_layers/", mute=False, back_memory=args.back_memory,
                                                               mem_divisor=args.mem_divisor, basic=args.basic))
        if args.layers in ["fc", "both"]:
            sub_accuracies_continuous.extend(train_layers(network, layers=[6], iterations=epochs_fc, metapochs=[
                                             3], mutes=[False], back_memory=args.back_memory, path="./nns/alexnet_layers/", mem_divisor=args.mem_divisor, basic=args.basic))

        accuracy = network.test_model()
        network.reinit_model()

        with open(f"alexnet_results_backmem={args.back_memory}_basic={args.basic}_layers={args.layers}_mem_divisor={args.mem_divisor}.txt", "a") as handle:
            handle.write(str(accuracy) + " ")

        with open(f"alexnet_results_backmem={args.back_memory}_basic={args.basic}_layers={args.layers}_mem_divisor={args.mem_divisor}_cont.txt", "a") as handle:
            for res in sub_accuracies_continuous:
                handle.write(str(res) + " ")
            handle.write("\n")


def call_lenet(args):
    """Calls learning of weights by CGP for Lenet model.

    Args:
        args (Namespace): Args from argparse.
    """
    accuracies_continuous = []
    network = Network_Loader("lenet", "./nns/lenet.pkl")
    print(network.model)
    print("Started training convolutional layers.")
    for i in range(args.runs):
        print(f"Run {i} started on LeNet")
        epochs_conv = [args.epochs_conv,
                       args.epochs_conv // 2, args.epochs_conv // 2]
        sub_accuracies_continuous = train_conv_layers(network, epochs=epochs_conv, layers=[0, 3, 5], metapochs=[5, 3, 3],
                                                      path="./nns/lenet_cgp/", mute=False, back_memory=args.back_memory,
                                                      mem_divisor=args.mem_divisor, basic=args.basic)
        accuracy = network.test_model()
        accuracies_continuous.append(sub_accuracies_continuous)
        with open(f"lenet_results_backmem={args.back_memory}_basic={args.basic}_layers={args.layers}_mem_divisor={args.mem_divisor}.txt", "a") as handle:
            handle.write(str(accuracy) + " ")
        with open(f"lenet_results_backmem={args.back_memory}_basic={args.basic}_layers={args.layers}_mem_divisor={args.mem_divisor}_cont.txt", "a") as handle:
            for res in sub_accuracies_continuous:
                handle.write(str(res) + " ")
            handle.write("\n")
        network.reinit_model()


if __name__ == "__main__":
    parser = add_arguments()

    args = parser.parse_args()

    print("Backward memory is set: ", args.back_memory)
    print("Basic functions are turn on: ", args.basic)
    print("Memory divisor is set to: ", args.mem_divisor)
    print("Selected network is: ", args.network)
    if args.network == "alexnet":
        call_alexnet(args)
    elif args.network == "lenet":
        call_lenet(args)
