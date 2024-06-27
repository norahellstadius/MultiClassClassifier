import os
import numpy as np
import splitfolders
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Dict, Any

from utils import check_dir
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization


def prepare_data_split(input_dir: str, output_dir: str) -> Tuple[str, str, str]:

    splitfolders.ratio(
        input_dir,
        output=output_dir,
        seed=1337,
        ratio=(0.8, 0.1, 0.1),
        group_prefix=None,
        move=False,
    )

    dir_train = os.path.join(output_dir, "train")
    dir_test = os.path.join(output_dir, "test")
    dir_val = os.path.join(output_dir, "val")
    return dir_train, dir_val, dir_test


def datagen(
    dir_train: str,
    dir_val: str,
    dir_test: str,
    batchSize: int = 32,
    preproc_func: Callable = tf.keras.applications.resnet.preprocess_input,
    pixel: int = 224,
    class_mode: str = "categorical",
) -> Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]:
    """
    Implement data loading using ImageDataGenerator & flow_from_directory.

    Arguments:
    batchSize -- the number of sets of images loaded at a time (default is 32)
    preproc_func -- the type of input preprocessing based on the specific model (default is ResNet preprocess_input)
    pixel -- the pixel dimensions (default is 224)
    class_mode -- mode for yielding the targets (default is 'binary')

    Returns:
    train_gen -- generator for training set.
    val_gen -- generator for validation set.
    test_gen -- generator for test set.
    """
    datagenerator = ImageDataGenerator(
        preprocessing_function=preproc_func,
        shear_range=0.2,
        zoom_range=0.3,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )

    train_gen = datagenerator.flow_from_directory(
        dir_train,
        target_size=(pixel, pixel),
        class_mode=class_mode,
        batch_size=batchSize,
        shuffle=True,
    )
    val_gen = datagenerator.flow_from_directory(
        dir_val,
        target_size=(pixel, pixel),
        class_mode=class_mode,
        batch_size=batchSize,
        shuffle=True,
    )
    test_gen = datagenerator.flow_from_directory(
        dir_test,
        target_size=(pixel, pixel),
        class_mode=class_mode,
        batch_size=batchSize,
        shuffle=True,
    )

    return train_gen, val_gen, test_gen


def get_class_weights(train_gen: DirectoryIterator) -> Dict[int, float]:
    """
    Calculate class weights to handle class imbalance.

    Arguments:
    train_gen -- generator for the training set

    Returns:
    class_weights_dict -- dictionary with class weights
    """
    classes = train_gen.classes
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(classes), y=classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    return class_weights_dict


def plot_history(model_history: Any, name: str, save_dir: str = "../plots/") -> None:
    """
    Plot and save the training history of the model.

    Arguments:
    model_history -- history object from model training
    name -- name of the file to save the plot
    save_dir -- directory to save the plot (default is "../plots/")
    """
    check_dir(save_dir)
    save_path = os.path.join(save_dir, f"{name}.png")
    plt.plot(model_history.history["accuracy"], label="accuracy")
    plt.plot(model_history.history["val_accuracy"], label="val_accuracy")
    plt.plot(model_history.history["loss"], label="loss")
    plt.plot(model_history.history["val_loss"], label="val_loss")
    plt.title("Model accuracy and loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy/Loss")
    plt.legend()
    plt.savefig(save_path)
    print(f"Training history plot is saved at: {save_path}")
    plt.close()


def create_model(
    train_set: DirectoryIterator,
    val_set: DirectoryIterator,
    class_weights: Dict[int, float],
    num_epochs: int = 5,
    path_to_save: str = "../models",
) -> Tuple[Sequential, Any]:
    """
    Create, train, and save a model using ResNet50 as the base.

    Arguments:
    train_set -- generator for the training set
    val_set -- generator for the validation set
    class_weights -- dictionary with class weights
    path_to_save -- directory to save the model (default is "../models")

    Returns:
    model -- trained model
    history -- training history object
    """

    check_dir(path_to_save)
    num_classes = len(np.unique(train_set.classes))
    model_res50 = ResNet50(
        weights="imagenet", include_top=False, input_shape=train_set.image_shape
    )

    # Create a Sequential model and add the ResNet50 layers
    model = Sequential()
    model.add(model_res50)

    # Freeze the ResNet50 layers
    for layer in model.layers:
        layer.trainable = False

    # Add new dense layers
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation="softmax"))

    optimizer = Adam(learning_rate=0.001)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    # Setting callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    my_callbacks = [early_stopping]

    # Train the model
    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=num_epochs,
        callbacks=my_callbacks,
        class_weight=class_weights,
    )
    model_save_path_pre = os.path.join(path_to_save, "modelResNet_50_pre.keras")
    model.save(model_save_path_pre)
    print(f"Model run for {num_epochs} epochs saved at: {model_save_path_pre}")

    for layer in model.layers:
        layer.trainable = True

    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=num_epochs + 3,
        initial_epoch=num_epochs,
        callbacks=my_callbacks,
        class_weight=class_weights,
    )
    model_save_path_post = os.path.join(path_to_save, "modelResNet_50_post.keras")
    model.save(model_save_path_post)
    print(f"Model run for {num_epochs + 3} epochs saved at: {model_save_path_pre}")

    return model, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fintune a model using ResNet50.")
    parser.add_argument(
        "--input_data_dir", type=str, default="../data", help="Directory where data is"
    )
    parser.add_argument(
        "--split_data_dir",
        type=str,
        default="../data",
        help="Directory where the split data is to be saved",
    )
    parser.add_argument(
        "--pixel", type=int, default=224, help="Pixel dimensions for the input images"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )

    args = parser.parse_args()
    dir_train, dir_val, dir_test = prepare_data_split(
        args.input_data_dir, args.split_data_dir
    )
    train_gen, val_gen, test_gen = datagen(
        dir_train=dir_train,
        dir_val=dir_val,
        dir_test=dir_test,
        batchSize=args.batch_size,
        pixel=args.pixel,
    )
    class_weights = get_class_weights(train_gen)
    model_res50_v2, Res50hist_v2 = create_model(
        train_set=train_gen,
        val_set=val_gen,
        class_weights=class_weights,
        num_epochs=args.epochs,
    )
    plot_history(Res50hist_v2, name="train_history")
