# MultiClassClassifier

Author: Nora Hallqvist

## Description 
This repository contains code to train/finetune and evaluate a multiclass classifier with a ResNet50 backbone.

## Application 
The classifier is designed to classify images into multiple predefined categories.

## Setup 

### 1. Clone the Repository

```bash
git clone git@github.com:norahellstadius/MultiClassClassifier.git
```

### 2. Set Up the Environment 

#### a. Create a Virtual Environment

**Note**: Make sure it is named `.env` (including the dot) since the `.gitignore` file tracks this exact name.

```bash
python3 -m venv .env
```

#### b. Activate the Virtual Environment

```bash
source .env/bin/activate
```

#### c. Install Requirements

With your virtual environment active, install the required packages using pip.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage 

### 1. Place Data in the Following Structure

```
data/
├── label_1
│   ├── image1.png
│   ├── image2.png
│   ├── image3.png
├── label_2
│   ├── image4.png
│   ├── image5.png
│   ├── image6.png
├── label_3
│   ├── image4.png
│   ├── image5.png
│   ├── image6.png
├── label_4
│   ├── image7.png

```

Here, `label_1`, `label_2`, ... can be any name associated with that class. 

### 2. Train the Model

#### a. Navigate to the Source Directory

```bash 
cd src
```

#### b. Run the Training Script

Command syntax: 
```bash
python3 train.py --input_data_dir <INPUT_DATA_DIR> --split_data_dir <SPLIT_DATA_DIR> [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--pixel PIXEL]
```

##### Required Arguments

- `--input_data_dir`: The path to the directory where the data is located.
- `--split_data_dir`: The path to the directory where you want the split data (train, val, test) to be stored.

##### Optional Arguments

- `--batch_size`: Batch size (default is 32).
- `--epochs`: The number of epochs for training (default is 10).
- `--pixel`: The pixel size of the images (default is 224).

For example, to run the training script with the default batch size (32), pixel size (224), and number of epochs (10), use the following command:

```bash 
python3 train.py --input_data_dir "PATH/TO/DATA" --split_data_dir "PATH/TO/SAVE/SPLIT/DATA"
```

After training is complete, the paths where the models are saved will be printed in the terminal. 
Two models are saved: a preliminary model (pre-model) and another model trained for an additional 3 epochs with more layers allowed to be trained (post-model).

The training loss plot is saved in `plots/train_history.png`. The directory is also printed in the terminal.

### 3. Test the Model

Stay in the `src` directory.

Command syntax: 
```bash
python3 test.py --test_dir <TEST_DIR> [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--pixel PIXEL]
```

##### Required Arguments

- `--test_dir`: The path to the directory where the test set is located.

##### Optional Arguments

- `--batch_size`: Batch size (default is 32).
- `--epochs`: The number of epochs for training (default is 10).
- `--pixel`: The pixel size of the images (default is 224).

For example, to run the test script with the default batch size (32) and pixel size (224), use the following command:

```bash 
python3 test.py --test_dir "PATH/TO/DATA/SPLIT/test" 
```

#### Evaluation Metrics calculated and saved

- A confusion matrix is saved for both models.
- Accuracy, AUC and how many images from each category is missclassified is printed in the terminal.
- Images that are false positives and false negatives are saved.

