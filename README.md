# Comparison of Anomaly Detection Methods in Chest X-rays

## Overview
This project, conducted as part of the INF8175 course at Polytechnique Montréal, compares several anomaly detection 
methods in the context of medical imaging. Inspired by the paper 
*Anomaly Detection in Medical Imaging with Deep Perceptual Autoencoders*, the project focuses on Deep Isolation Forest 
(Deep IF), Perceptual Image Anomaly Detection (PIAD), and Deep Perceptual Autoencoder (DPA) methods applied to chest 
X-ray datasets. The evaluation includes metrics such as ROC-AUC, accuracy, and F1 score, as well as the analysis of 
confusion matrices. The project compares its results with those reported in the original paper, 
crediting the code repository where implementations were utilized.

## Contents
The project is organized as follows:
1. **notebooks**: Contains all the notebooks used to launch the various training sessions and analyzes them from Google Colab.
2. **anomaly_detection**: Package containing the implementation of the various models used (see **Credits**).
3. **data**: Useful data, including a subset of [ChestX-ray14](https://paperswithcode.com/dataset/chestx-ray14) of 
over 7,000 images resized to 300x300.
4. **Comparison_of_Anomaly_Detection_Methods_in_Chest_X-rays.pdf** : Our “article” contains an in-depth analysis of our results and comparisons between the different models.

## How to Use
This project is designed to be executed from notebooks, originally intended for Google Colab. To use:
1. Clone the repository to your local machine.
2. Open and run the provided notebooks in your preferred environment.
3. Follow instructions provided in the notebooks to execute the code and reproduce experiments.

## Contributors
- [Léo Valette](https://github.com/deca1111)
- [Daniel Alejandro Galindo Lazo](https://github.com/danigl00)
- [Isabel Sarzo Wabi](https://github.com/isabelsarzo)

## Credits
The code implementations utilized in this project were sourced from 
[this repository](https://github.com/ninatu/anomaly_detection). 

See the original paper for more information: [Anomaly Detection in Medical Imaging With Deep Perceptual Autoencoders](https://arxiv.org/abs/2006.13265)   
by Shvetsova, Nina and Bakker, Bart and Fedulova, Irina and Schulz, Heinrich and Dylov, Dmitry V.