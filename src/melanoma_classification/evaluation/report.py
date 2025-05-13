import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def visualize_confusion_matrix(
    evaluation_report: pd.DataFrame, class_labels: list
) -> plt.Figure:
    """Create the confusion matrix (incl. total mount & probablities) from the
    evaluation results and visualize/save it.

    Args:
        evaluation_report: DataFrame containing predicted and true labels.
        class_labels: List of class labels.
    """
    true_labels = evaluation_report["true_class"]
    predicted_labels = evaluation_report["predicted_class"]

    cm = confusion_matrix(true_labels, predicted_labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 7))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    # Annotate with absolute numbers and probabilities
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                ha="center",
                va="center",
                color="red",
                fontsize=12,
            )
    accuracy = cm.diagonal().sum() / cm.sum()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - Accuracy: {accuracy*100:.2f}")

    return fig


def visualize_model_confidence(evaluation_report: pd.DataFrame) -> plt.Figure:
    """
    Create a box plot visualization of model confidence based on the evaluation
    DataFrame using matplotlib.

    Args:
        evaluation_report: DataFrame containing predicted and true labels.
    """
    true_positive = evaluation_report[
        (evaluation_report["predicted_class"] == 0)
        & (evaluation_report["true_class"] == 0)
    ]

    true_negative = evaluation_report[
        (evaluation_report["predicted_class"] == 1)
        & (evaluation_report["true_class"] == 1)
    ]

    false_positive = evaluation_report[
        (evaluation_report["predicted_class"] == 0)
        & (evaluation_report["true_class"] == 1)
    ]

    false_negative = evaluation_report[
        (evaluation_report["predicted_class"] == 1)
        & (evaluation_report["true_class"] == 0)
    ]

    # Prepare the data for plotting
    data = [
        true_positive["predicted_confidence"],
        true_negative["predicted_confidence"],
        false_positive["predicted_confidence"],
        false_negative["predicted_confidence"],
    ]

    labels = ["TP", "TN", "FP", "FN"]

    # Create the box plots
    fig = plt.figure(figsize=(10, 6))
    plt.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
    )
    plt.title("Model Confidence for Predictions (Positives=Melanoma)")
    plt.ylabel("Confidence")
    plt.grid(True, axis="y")

    plt.tight_layout()

    return fig


def create_evaluation_report(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """Create the evaluation function for the model

    It returns the pandas DataFrame with the results of the evaluation,
    i.e. predicted labels & confidence and true labels.

    Args:
        model: VisionTransformer model
        test_dataloader: DataLoader for the test dataset
        device: Device to run the model on

    Returns:
        DataFrame containing predicted and true labels.
    """
    model.eval()
    results = []
    for image, label, id in tqdm(dataloader, unit="Images"):
        id = id[0]
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = model(image)["outputs"]
        _, predicted = torch.max(output, 1)
        results.append(
            {
                "image": id,
                "true_class": label.item(),
                "predicted_class": predicted.item(),
                "predicted_confidence": torch.nn.functional.softmax(
                    output, dim=1
                )[0][predicted.item()].item(),
            }
        )
    return pd.DataFrame(results)
