import argparse
import yaml
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from anomaly_detection.piad.train import Trainer
from anomaly_detection.utils.datasets import DatasetType, DATASETS
from anomaly_detection.utils.transforms import TRANSFORMS


def evaluate(config):
    batch_size = config['test_batch_size']
    results_root = config['results_root']
    model_path = config['test_model_path']

    os.makedirs(results_root, exist_ok=True)

    # print(yaml.dump(config, default_flow_style=False))

    print("Starting model evaluation ...")

    enc, gen, image_rec_loss, niter = \
        Trainer.load_anomaly_detection_model(torch.load(model_path))
    enc, gen, image_rec_loss = enc.cuda().eval(), gen.cuda().eval(), image_rec_loss.cuda().eval()

    dataset_type = config['test_datasets']['normal']['dataset_type']
    dataset_kwargs = config['test_datasets']['normal']['dataset_kwargs']
    transform_kwargs = config['test_datasets']['normal']['transform_kwargs']

    transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
    normal_dataset = DATASETS[DatasetType[dataset_type]](
        transform=transform,
        **dataset_kwargs
    )

    dataset_type = config['test_datasets']['anomaly']['dataset_type']
    dataset_kwargs = config['test_datasets']['anomaly']['dataset_kwargs']
    transform_kwargs = config['test_datasets']['anomaly']['transform_kwargs']
    transform = TRANSFORMS[DatasetType[dataset_type]](**transform_kwargs)
    anomaly_dataset = DATASETS[DatasetType[dataset_type]](
        transform=transform,
        **dataset_kwargs
    )

    norm_anomaly_scores = predict_anomaly_scores(gen, enc, image_rec_loss, normal_dataset, batch_size)
    an_anomaly_scores = predict_anomaly_scores(gen, enc, image_rec_loss, anomaly_dataset, batch_size)

    y_true = np.concatenate((np.zeros_like(norm_anomaly_scores), np.ones_like(an_anomaly_scores)))
    y_pred = np.concatenate((np.array(norm_anomaly_scores), np.array(an_anomaly_scores)))

    print(y_true)
    print(y_pred)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # get best treshold
    bestScore = -10
    bests = None
    for index in range(len(roc_thresholds)):
      score = tpr[index] - fpr[index]
      if score > bestScore:
        bestScore = score
        bests = (fpr[index], tpr[index], roc_thresholds[index])

    print(bests)

    plt.figure(figsize=(8, 6))

	  # Plot ROC curve
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    # Plot diagonal
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # plot best treshold
    plt.scatter(bests[0], bests[1], s=100, lw=2, color='r', edgecolors="k", label=f"Best threshold = {bests[2]:0.3f}", zorder=2)


    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Deep IF ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

	  # Calculate F1 score
    f1 = f1_score(y_true, y_pred>bests[2])

    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred>bests[2])

    output_path = os.path.join(results_root, 'results.csv')
    results = pd.DataFrame([[niter, roc_auc, acc, f1]], columns=['niter', 'ROC AUC', 'Accuracy', 'F1'])

    print("Model evaluation is complete. Results: ")
    print(results)
    results.to_csv(output_path, index=False)

	# Save ROC curve data to a text file
    data = np.column_stack((y_true, y_pred))
    np.savetxt('PIAD_Results.txt', data, fmt='%.6f', delimiter=',', header='y_true, y_pred')


def predict_anomaly_scores(gen, enc, image_rec_loss, dataset, batch_size):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    data_loader = tqdm(data_loader)

    image_rec_loss.set_reduction('none')

    anomaly_scores = []
    for images in data_loader:
        images = images.cuda()
        with torch.no_grad():
            rec_images = gen(enc(images)).detach()
            cur_anomaly_scores = image_rec_loss(images, rec_images)
        anomaly_scores.extend(cur_anomaly_scores.detach().cpu().numpy())

    return anomaly_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', type=str, nargs='*', help='Config paths')

    args = parser.parse_args()

    for config_path in args.configs:
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader)

        evaluate(config)


if __name__ == '__main__':
    main()
