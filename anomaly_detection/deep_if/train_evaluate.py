import os
import tqdm

import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import sklearn.ensemble
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import torch

from anomaly_detection.deep_if.feature_extractor import InceptionResNetV2FeatureExtractor
from anomaly_detection.utils.datasets import DATASETS, DatasetType
from anomaly_detection.utils.transforms import TRANSFORMS


def _load_dataset(dataset_type, dataset_kwargs, transform_kwargs):
    transform = TRANSFORMS[DatasetType(dataset_type)](**transform_kwargs, to_tensor=False, normalize=False)
    is_grayscale = True if (dataset_type == 'nih') or (transform_kwargs.get('to_grayscale', 'False') is True) else False
    transform = transforms.Compose(
        [transform,
        transforms.Resize((299, 299)),
        transforms.ToTensor()] +
        ([transforms.Lambda(lambda x: x.repeat(3, 1, 1))] if is_grayscale else []) +
        [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = DATASETS[DatasetType(dataset_type)](**dataset_kwargs, transform=transform)
    return dataset


def _get_features(feature_extractor, dataset, batch_size):
    torch.set_grad_enabled(False)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                             num_workers=8, pin_memory=False)
    feature_extractor.eval().cuda()

    features = []

    for batch in tqdm.tqdm(data_loader):
        features.extend(feature_extractor(batch.cuda()).cpu().numpy())
    torch.set_grad_enabled(False)

    return features


def train_evaluate(config):
    random_seed = config['random_seed']
    results_root = config['results_root']
    os.makedirs(results_root, exist_ok=True)

    batch_size = config['batch_size']
    feature_extraction_type = config['feature_extraction_type']

    train_dataset = _load_dataset(**config['train_dataset'])
    test_normal_dataset = _load_dataset(**config['test_datasets']['normal'])
    test_anomaly_dataset = _load_dataset(**config['test_datasets']['anomaly'])

    "================================= training ====================================="

    print("Starting model training ...")

    if random_seed is not None:
        np.random.seed(random_seed)

    feature_extractor = InceptionResNetV2FeatureExtractor(feature_extraction_type)
    feature_extractor.eval().cuda()

    train_features = _get_features(feature_extractor, train_dataset, batch_size)

    model = sklearn.ensemble.IsolationForest()
    model.fit(train_features)

    del train_features

    print("Model training is complete.")

    "================================= evaluation ====================================="

    print("Starting model evaluation ...")

    normal_features = _get_features(feature_extractor, test_normal_dataset, batch_size)
    normal_scores = model.score_samples(normal_features)
    del normal_features

    anomaly_features = _get_features(feature_extractor, test_anomaly_dataset, batch_size)
    anomaly_scores = model.score_samples(anomaly_features)
    del anomaly_features

    labels = np.concatenate((np.ones(normal_scores.shape), np.zeros(anomaly_scores.shape)))
    scores = np.concatenate((normal_scores, anomaly_scores))

    fpr, tpr, roc_thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # Compute F1 score
    threshold = 0
    y_pred = (scores < threshold).astype(int)
    f1 = f1_score(labels, y_pred)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()

    # Plot ROC curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Deep IF ROC Curve')
    plt.legend(loc="lower right")

    # Plot confusion matrix
    plt.subplot(1, 2, 2)
    cm = np.array([[tp, fp], [fn, tn]])
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Anomaly', 'Normal'], rotation=45)
    plt.yticks(tick_marks, ['Anomaly', 'Normal'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    plt.show()

    results = pd.DataFrame([[roc_auc, f1]], columns=['ROC AUC', 'F1 Score'])
    print("Confusion Matrix:")
    print(cm)
    print("\nModel evaluation is complete. Results: ")
    print(results)
    results.to_csv(os.path.join(results_root, 'results.csv'))