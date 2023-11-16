
# Flair NER Model Training

[![Flair NLP](https://github.com/flairNLP/flair/blob/master/resources/docs/flair_logo_2020_FINAL_day_dpi72.png#gh-light-mode-only)](https://github.com/flairNLP/flair)

This repository contains scripts for training Named Entity Recognition (NER) models using Flair, a powerful NLP framework. The training process is streamlined and configurable, allowing for easy experimentation with different model architectures and training parameters.

## Features

- **Configurable Training**: Customize various aspects of the training process such as model selection, learning rate, batch size, and more through a simple configuration file.
- **Automated Script Execution**: Bash scripts for easy training execution.
- **Logging Integration**: Enhanced logging for better tracking of the training process.
- **Error Handling**: Robust error handling mechanisms to ensure smooth training runs.

## Configuration

Training configurations are managed via `config.ini`. Here's a brief overview of the configurable parameters:

```ini
[NER_Training]
models = [Model Names]
learning_rate = [Learning Rate]
max_epochs = [Max Epochs]
mini_batch_size = [Mini Batch Size]
output_folder = [Path to Save Models]
fine_tune = [True/False]
fine_tune_layers = [Fine Tune Layers]
```

For a detailed explanation of each parameter, refer to the comments in `config.ini`.

## Training Process

The training process involves several stages, all handled by the `FlairNERTrainer` class:

1. **Data Loading**: Load and preprocess the training data.
2. **Model Initialization**: Configure and initialize the model based on specified parameters.
3. **Model Training**: Train the model with the provided corpus.
4. **Model Saving**: Save the trained model to the specified directory.

## Running the Training

To train your NER model, follow these steps:

1. **Set Up Your Environment**: Ensure Flair and its dependencies are installed in your Python environment (should already be okay if you used poetry to install this repo).

2. **Configure Training Parameters**: Edit `config.ini` to set your desired training parameters.

3. **Run the Bash Script**: Execute `train_ner.sh` to start the training process:

   ```bash
   ./train_ner.sh
   ```

## Results

The trained models will be saved in the directory specified in the `config.ini`. Each model will be stored in a separate subfolder named after the model and its training parameters for easy identification.


# Flair Relation ClassificationModel Training

This repository provides scripts for training Relation Extraction models using the Flair framework. The process is designed to be highly configurable, facilitating experimentation with various models and training configurations.

## Features

- **Configurable Training Setup**: Adjust model architecture, learning rate, batch size, and more through a configuration file.
- **Automated Training Execution**: Use bash scripts to streamline the training process.
- **Enhanced Logging**: Integrated logging for better monitoring of the training process.
- **Robust Error Handling**: Error handling mechanisms to ensure reliable training execution.

## Configuration

The training parameters are managed via `config.ini`. Key configurable parameters include:

```ini
[RelationExtraction]
models = [List of Model Names]
learning_rate = [Learning Rate]
max_epochs = [Maximum Epochs]
mini_batch_size = [Mini Batch Size]
output_folder = [Path for Model Output]
fine_tune = [True/False for Fine Tuning]
fine_tune_layers = [Layers to Fine Tune]
entity_label_map_path = [Path to Entity Label Map File]
```

Refer to `config.ini` for a comprehensive explanation of each parameter.

## Training Workflow

The `FlairRelationTrainer` class automates the following stages:

1. **Corpus Preparation**: Loads and sets up the training corpus.
2. **Model Configuration**: Initializes the model based on the specified settings.
3. **Training Execution**: Conducts the model training with the given corpus.
4. **Model Preservation**: Saves the trained models in the designated directory.

### This is what your training files should look like if you are using the CONLL-U format.

```
# text = Ik zie in mijn mijnkpn zakelijk dat ik kennelijk 2 dezelfde abonnementen heb en ik betaal dus ook dubbel, terwijl dat niet het geval is.
# relations = 8;8;12;12;heb|1;1;5;5;interactie
1	Ik	S-PERSON
2	zie	O
3	in	O
4	mijn	O
5	mijnkpn	S-DIEPROFUN
6	zakelijk	O
7	dat	O
8	ik	S-PERSON
9	kennelijk	O
10	2	O
11	dezelfde	O
12	abonnementen	S-DIEPROFUN
13	heb	O
14	en	O
15	ik	S-PERSON
16	betaal	O
17	dus	O
18	ook	O
19	dubbel	O
20	,	O
21	terwijl	O
22	dat	O
23	niet	O
24	het	O
25	geval	O
26	is	O
27	.	O
```

## How to Train Your Model

Follow these steps to start the training:

1. **Prepare Your Python Environment**: Make sure that Flair and necessary dependencies are installed (again, with preferably using poetry).

2. **Set Training Parameters**: Edit `config.ini` to specify your training preferences.

3. **Execute the Bash Script**: Run `train_relation.sh` to initiate the training:

   ```bash
   ./train_rel.sh
   ```


