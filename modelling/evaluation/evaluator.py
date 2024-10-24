import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from melanoma_classification.model import VisionTransformer
from sklearn.metrics import confusion_matrix

def visualize_loss(
    df_metrics: pd,
    save_path: Path = None
) -> None:
    """Visualize training and validation loss.
    
    Args:
        df_metrics: DataFrame containing training and validation
        loss.
        save_path: Path to save the plot.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_metrics['epoch'], df_metrics['train_loss'], label='Train Loss')
    ax.plot(df_metrics['epoch'], df_metrics['val_loss'], label='Validation Loss')
    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_f1_precision_recall(
    df_metrics: pd,
    save_path: Path = None
) -> None:
    """Visualize F1, precision, and recall in same plot for training and
    validation.

    Args:
        df_metrics: DataFrame containing training and validation
        loss.
        save_path: Path to save the plot.
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_metrics['epoch'], df_metrics['train_f1'], label='Train F1')
    ax.plot(df_metrics['epoch'], df_metrics['train_precision'], label='Train Precision')
    ax.plot(df_metrics['epoch'], df_metrics['train_recall'], label='Train Recall')
    ax.plot(df_metrics['epoch'], df_metrics['val_f1'], label='Validation F1')
    ax.plot(df_metrics['epoch'], df_metrics['val_precision'], label='Validation Precision')
    ax.plot(df_metrics['epoch'], df_metrics['val_recall'], label='Validation Recall')
    ax.set_title('Training and Validation Metrics')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_accuracy(
    df_metrics: pd,
    save_path: Path = None
) -> None:
    """Visualize accuracy in same plot for training and validation.

    Args:
        df_metrics: DataFrame containing training and validation
        loss.
        save_path: Path to save the plot.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_metrics['epoch'], df_metrics['train_acc'], label='Train Accuracy')
    ax.plot(df_metrics['epoch'], df_metrics['val_acc'], label='Validation Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def create_evaluation_report(
    model: VisionTransformer,
    test_dataloader: torch.utils.data.DataLoader,
    class_labels: list,
    device: torch.device
) -> pd.DataFrame:
    """Create the evaluation function for the model

    It returns the pandas DataFrame with the results of the evaluation,
    i.e. predicted labels & confidence and true labels.

    class labels = ['Melanoma', 'NotMelanoma']

    Args:
        model: VisionTransformer model
        test_dataloader: DataLoader for the test dataset
        class_labels: List of class labels
        device: Device to run the model on
    """
    model.eval()
    results = []
    for i, (image, label) in enumerate(test_dataloader):
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = model(image)["outputs"]
        _, predicted = torch.max(output, 1)
        model.classes
        results.append({
            'predicted_class': predicted.item(),
            'predicted_label': class_labels[predicted.item()],
            'predicted_confidence': torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item(),
            'true_class': label.item(),
            'true_label': class_labels[label.item()]
        })
    return pd.DataFrame(results)


def visualize_confusion_matrix(
    evaluation_report: pd.DataFrame,
    class_labels: list,
    save_path: Path = None
) -> None:
    """Create the confusion matrix (incl. total mount & probablities) from the
    evaluation results and visualize/save it.

    Args:
        evaluation_report: DataFrame containing predicted and true labels.
        class_labels: List of class labels.
        save_path: Path to save the plot
    """
    true_labels = evaluation_report['true_class']
    predicted_labels = evaluation_report['predicted_class']
    
    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 7))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    
    # Set axis labels
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    
    # Annotate with absolute numbers and probabilities
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                    ha='center', va='center', color='red', fontsize=12)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix with Absolute Numbers and Probabilities')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_model_confidence(
    evaluation_report: pd.DataFrame,
    save_path: Path = None
) -> None:
    """
    Create a box plot visualization of model confidence based on the evaluation
    DataFrame using matplotlib.

    Args:
        evaluation_report: The evaluation DataFrame containing
        'predicted_class', 'predicted_label', 'predicted_confidence',
        'true_class', and 'true_label'.
    """
    
    # Define the conditions for each category
    true_positive = evaluation_report[(evaluation_report['predicted_class'] == 0) & (evaluation_report['true_class'] == 0)]
    true_negative = evaluation_report[(evaluation_report['predicted_class'] == 1) & (evaluation_report['true_class'] == 1)]
    false_positive = evaluation_report[(evaluation_report['predicted_class'] == 0) & (evaluation_report['true_class'] == 1)]
    false_negative = evaluation_report[(evaluation_report['predicted_class'] == 1) & (evaluation_report['true_class'] == 0)]
    
    # Prepare the data for plotting
    data = [
        true_positive['predicted_confidence'],
        true_negative['predicted_confidence'],
        false_positive['predicted_confidence'],
        false_negative['predicted_confidence']
    ]
    
    labels = ['TP', 'TN', 'FP', 'FN']

    # Create the box plots
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    plt.title('Model Confidence for Predictions (Positives=Melanoma)')
    plt.ylabel('Confidence')
    plt.grid(True, axis='y')
    
    # Show the plot
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()