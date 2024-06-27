import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from utils import check_dir

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def datagen(test_dir: str, batchSize: int = 32, pixel: int = 224):
    """
    Implement data loading using ImageDataGenerator & flow_from_directory
    --------------------------------------------------------------------------
    Arguments:
    pixel: the pixel dimensions
    batchSize: the number of set of images loaded at a time
    --------------------------------------------------------------------------
    Returns:
    test_gen -- variable storing test set.
    """
    datagenerator = ImageDataGenerator()
    test_gen = datagenerator.flow_from_directory(
        test_dir,
        target_size=(pixel, pixel),
        class_mode="categorical",
        batch_size=batchSize,
        shuffle=False,
    )
    return test_gen


def evaluate_model(
    dir_test: str, model, test_gen: DirectoryIterator, output_dir: str
) -> Tuple[float, float, int, int, int, int]:
    """
    Evaluate a model by predicting on a test dataset, calculating metrics, and saving misclassified images.

    Args:
        dir_test (str): Directory containing the test images.
        model: Trained model to evaluate.
        test_gen (DirectoryIterator): Test data generator.
        output_dir (str): Directory to save the evaluation results.

    Returns:
        Tuple[float, float, int, int, int, int]: Accuracy, mean AUC, false positives, false negatives, true negatives, true positives.
    """
    check_dir(output_dir)

    y_pred_prob = model.predict(test_gen)
    y_pred = np.argmax(y_pred_prob, axis=1)  # Predicted class labels
    y_true = test_gen.classes  # True class labels
    class_labels = list(
        test_gen.class_indices.keys()
    )  # Get class labels from generator

    output_dir_misclass = os.path.join(output_dir, "misclassified")
    check_dir(output_dir_misclass)

    # Identify misclassified images and copy them
    misclassified_counts = {label: 0 for label in class_labels}
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            src = os.path.join(dir_test, test_gen.filenames[i])
            dst = os.path.join(
                output_dir_misclass,
                f"true_{class_labels[y_true[i]]}_pred_{class_labels[y_pred[i]]}_{os.path.basename(test_gen.filenames[i])}",
            )
            shutil.copyfile(src, dst)
            print(
                f"Copied misclassified image: {test_gen.filenames[i]} to {output_dir_misclass}"
            )
            misclassified_counts[class_labels[y_true[i]]] += 1

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(
        y_true, y_pred_prob, multi_class="ovr"
    )  # Calculate mean AUC for multiclass
    cm = confusion_matrix(
        y_true, y_pred, labels=np.arange(len(class_labels))
    )  # Confusion matrix

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    # Save confusion matrix plot
    cm_image_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_image_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_image_path}")

    return accuracy, auc, misclassified_counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the test set."
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="../data/split/test",
        help="Directory to the test set",
    )
    parser.add_argument(
        "--pixel", type=int, default=224, help="Pixel dimensions for the input images"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Pixel dimensions for the input images",
    )

    args = parser.parse_args()

    # Generate the test set
    test_gen = datagen(args.test_dir, batchSize=args.batch_size, pixel=args.pixel)

    # Load the pre-trained model
    model_pre = tf.keras.models.load_model("../models/modelResNet_50_pre.keras")
    # Load the fully trained model
    model_post = tf.keras.models.load_model("../models/modelResNet_50_post.keras")

    # Evaluate the pre-trained model
    accuracy_pre, auc_pre, misclassified_counts_pre = evaluate_model(
        args.test_dir, model_pre, test_gen, "../results/pre"
    )
    print(
        f"Pre-trained Model - Accuracy: {accuracy_pre:.4f}, AUC Score: {auc_pre:.4f}, Misclassfied Counts: {misclassified_counts_pre}"
    )

    # Evaluate the fully trained model
    accuracy_post, auc_post, misclassified_counts_post = evaluate_model(
        args.test_dir, model_post, test_gen, "../results/post"
    )
    print(
        f"Fully Trained Model - Accuracy: {accuracy_post:.4f}, AUC Score: {auc_post:.4f}, Misclassfied Counts: {misclassified_counts_post}"
    )
