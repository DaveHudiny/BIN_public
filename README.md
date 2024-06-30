# Solution of project to subject Biology Inspired Computers

Autor: David Hudák
Login: xhudak03
Academic year: 2022/2023
Topic: Weights Generators for CNNs

This folder contains source files of project, Makefile for installation, final presentation and file requirements.txt, which determines requirements for project.

## Installation

For installation, I recommend to use virtual environment with Python3.7 or Python3.8. With other versions I have met with some library compatibility problem (e.g. for numpy).

You can install required libraries and modules by calling make. Makefile will do the job.

## Sources

In this project I work with implementation of CGP algorithm from: https://github.com/Happy-Algorithms-League/hal-cgp

## Usage

Before you start using main script, you should (at least I recommend it) use:

$ python3 network_loader.py

This script will download neural network and train it for couple of epochs. In default, it loads lenet network only -- if you want to work with AlexNet, you have to change model_name="lenet" at line 157 to model_name="alexnet" (and save path similarly). This will load and train AlexNet network.

After you got model and you want to use cgp for weight approximation by cgp, just call python3 main.py -h. It will tell you how to use the script, which trains weights.

If you get bored, you can also use script starter.sh, which will start whole set of experiments and store it to current folder.
