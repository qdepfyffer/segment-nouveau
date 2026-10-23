# Segment-nouveau

**nouveau**  
*adjective*  
nou·veau - /nü-ˈvō/  
Definition: newly arrived or developed

A (hopefully more) modular framework for the implementation, training, and testing of various Semantic Segmentation models, rebuilt from the ground up for CIWA+ purposes. Should also be easy enough to modify for other purposes if you know what you're doing and aren't using this as part of the CIWA+ project. 

For reference: This repo was built based on [AIISLab segment-main](https://github.com/AIISLab/segment-main) repo, which is, as far as I can tell, the successor to [George Seif's Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite) repo that appears to have been used in the earlier CIWA+ paper.

### Important notes:
 - Running these scripts requires an NVIDIA GPU
 - Default crop size was chosen for the tomato dataset, other datasets will need this set to be specific to them
 - This project uses [hydra](https://hydra.cc/) now, rather than argparse. This changes the way command line arguments are input (`arg=val` instead of `--arg val`) and adds some new functionalities. It changes a few other things on the implementation side as well. These differences will be detailed further in this README.md when the transition to hydra is fully complete (and I've had time to write them all down).
 - When passing filepaths for the data_root parameter, use full paths; $HOME/... works, a bare ~/... doesn't.

## Dataset Requirements:  
Your dataset should follow the example structure below. In short: One directory containing directories for each crop, each with their own test, train, and valid folders, with each split having a visual and label folder. Extra folders are ignored.

Visual images are matched with their labels via their stems, where the labels have "_L" appended to their stem (i.e. the label for visual/img.jpg should be at label/img_L.png).

Labels use exactly two colors, (0, 255, 0) for sunlit leaves and (165, 42, 42) for anything that isn't part of a sunlit leaf.

Images and their labels must be the same dimensions.

*Example folder structure can be seen below. Assume "..." means "the same as the previous folder in this subdirectory."*

```
data_root/  
├── almond  
│   ├── test  
│   │   ├── label  
│   │   └── visual  
│   ├── train  
│   │   └── ...  
│   └── valid  
│       └── ...  
├── citrus  
│   └── ...  
├── grape  
│   └── ...  
├── pistachio  
│   └── ...  
└── tomato  
    └── ...
```

## Getting Started

*Segment-nouveau is built and tested with python 3.11*

### 1. Clone and enter the project

```bash
git clone https://github.com/qdepfyffer/segment-nouveau
cd segment-nouveau
```

### 2. Create and activate a virtual environment

**Linux/MacOS:**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

*If you use fish:*
```fish
python3.11 -m venv .venv
source .venv/bin/activate.fish
```

**Windows (PS):**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

> If PowerShell refuses to run the activation script, run
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then retry. *
> 
> \* https://stackoverflow.com/a/4038991

**Windows (CMD):**

```bat
py -3.11 -m venv .venv
.venv\Scripts\Activate.bat
```

### 3. Install requirements

*Must be done in this order, as some packages may depend on torch. Also note that Torch is pinned to a specific CUDA build - 12.6*

```bash
pip install -r requirements-torch.txt
pip install -r requirements.txt
```

### 4. Verify torch installation

```bash
python -m tools.check_env
```
Running the above command should print something like:
```
Torch version: 2.14.1+cu126
CUDA Available: True
Torch CUDA version: 12.6
CUDA device: NVIDIA GeForce RTX 3080
Tensor sum: 12.0
```

## Training a model

**Example usage**: `python train.py data_root=path/to/dataset crop=tomato epochs=100 patience=7 batch_size=8 lr=1e-4`

Checkpoints are saved to `checkpoints/crop/timestamp/architecture-best.pt` by default, along with the `settings.json` that was used to produce them and the `history.json` that accompanied their training.

### Command line arguments (Training)

| Argument       | Description                                                                                                                  | Example                  |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `data_root`    | Root dataset directory, should have a subdirectory for the crop you want to train on                                         | `/home/ciwa-full-split/` |
| `crop`         | The crop (name of subdirectory within the dataset root directory) you want to train on                                       | `tomato`                 |
| `architecture` | Which model to train                                                                                                         | `deeplabv3`              |
| `optimizer`    | Which optimizer to use during training                                                                                       | `rmsprop`                |
| `dice_weight`  | Determines dice weight to be used for loss - cross-entropy plus dice weight multiplied by dice loss - 0 to disable dice loss | `1.0`                    |
| `epochs`       | Maximum number of epochs to train for (passes over the training set)                                                         | `100`                    |
| `monitor`      | Which metric to monitor during training for early stopping and determining the best model to save                            | `valid_loss`             |
| `patience`     | How many epochs to continue training for without seeing improvement in the monitored value                                   | `7`                      |
| `batch_size`   | Number of images in each training batch                                                                                      | `8`                      |
| `lr`           | The learning rate of the model                                                                                               | `1e-4`                   |
| `train_crop`   | The size of the random crop that is applied to training images (h x w)                                                       | `256 480`                |
| `output_dir`   | The folder this script should output to                                                                                      | `checkpoints`            |
| `seed`         | Number to use as a random seed for anything that can be seeded                                                               | `42`                     |


## Evaluating a model

**Example usage**: `python evaluate.py --checkpoint checkpoints/tomato/2026-10-11_07-48-57/ --crop tomato --split test`

Evaluation results are saved to `eval/<run folder name>/<crop>-<split>.json` by default, where "crop" represents the evaluated dataset and "split" the evaluated split.

### Command line arguments (Evaluation)

| Argument     | Description                                                                                                                | Example                                   |
| ------------ | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `checkpoint` | Path to the folder containing the desired checkpoint and settings.json                                                     | `checkpoints/tomato/2026-10-11_07-48-57/` |
| `crop`       | The crop (name of subdirectory within the dataset root directory) you want to use in evaluation                            | `tomato`                                  |
| `split`      | The split the model should consider for evaluation                                                                         | `test`                                    |
| `output_dir` | The folder this script should output to                                                                                    | `eval/`                                   |
| `data_root`  | Root dataset directory override, for when the data_root in the model's settings.json does not exist on the current machine | `/home/ciwa-full-split/`                  |

## Extending the suite

### Adding a new model

*See `models/__init__.py`*

After adding and training a new model, run `python -m tools.check_sizing --checkpoint <run folder> --crop <crop>` for each crop you plan to report

### Adding a new loss function

*See `components/losses.py`*

### Adding a new optimizer

*See `components/optimizers.py`*

### Adding new transforms

*See `components/transforms.py`*

## Repo Contents
| File                        | Contents                                                                                                              |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `components/`               | `__init__.py`, `losses.py`, `metrics.py`, `optimizers.py`, `transforms.py`                                            |
| `components/__init__.py`    | Empty                                                                                                                 |
| `components/losses.py`      | Builder and implementations for the loss functions used during model training                                         |
| `components/metrics.py`     | Functions needed for the evaluation and scoring of models                                                             |
| `components/optimizers.py`  | Builder and implementations for the optimizers used during model training                                             |
| `components/transforms.py`  | Builder and implementations for any transforms used during model training or evaluation                               |
| `conf/`                     | `crop/`, `model/`, `check_sizing.yaml`, `evaluate.yaml`, `train.yaml`                                                 |
| `conf/crop/`                | `tomato.yaml`                                                                                                         |
| `conf/crop/tomato.yaml`     | Hydra config file for tomato specific settings                                                                        |
| `conf/model/`               | `deeplabv3.yaml`                                                                                                      |
| `conf/model/deeplabv3.yaml` | Hydra config file for DeepLabV3 specific settings                                                                     |
| `conf/check_sizing.yaml`    | Hydra config file for `check_sizing.py`                                                                               |
| `conf/evaluate.yaml`        | Hydra config file for `evaluate.py`                                                                                   |
| `conf/train.yaml`           | Hydra config file for `train.py`                                                                                      |
| `models/`                   | `__init__.py`, `deeplabv3.py`, `sizing.py`                                                                            |
| `models/__init__.py`        | Dictionary of available models and a model builder function                                                           |
| `models/deeplabv3.py`       | Implementation of the DeepLabV3 model with a MobileNetV3 backbone                                                     |
| `models/sizing.py`          | Function to pad a batch of images on the right/bottom via reflection so height and width cleanly divide by `multiple` |
| `tools/`                    | `__init__.py`, `check_env.py`, `check_sizing.py`                                                                      |
| `tools/__init__.py`         | Empty                                                                                                                 |
| `tools/check_env.py`        | Script to run a few commands to validate installation of PyTorch                                                      |
| `tools/check_sizing.py`     | Script to check whether or not a model's sizing is set correctly, and whether it can tolerate black padding           |
| `dataset.py`                | Finds images and their corresponding labels, loads them, ensures they match                                           |
| `evaluate.py`               | Evaluation logic and loop, this is what you run to evaluate the models                                                |
| `README.md`                 | What you're reading now.                                                                                              |
| `requirements.txt`          | Non-torch project requirements                                                                                        |
| `requirements-torch.txt`    | Torch-specific project requirements                                                                                   |
| `train.py`                  | Training logic and loop, this is what you run to train the models                                                     |

## Design decisions, or alternatively "Why would you rebuild a perfectly good* tool?"

 - Images are never stretched. If models expect inputs to have a specific aspect ratio, we instead take a randomly cropped window from our full images, so that the model isn't learning from a distorted view of our data. This follows the approach of the suite used in the earlier papers.
 - When a model is run, images are padded out to clean multiples of a size specified by the model implementation. This padding uses mirroring rather than black pixels. Never pad with black. With the default training crop behavior, this only affects validation and testing
 - Validation images handled with a batch size of 1. This is to allow for the differently sized images within the tomato dataset. Previous implementations stretched everything to 512x512, so never had batches of mixed sizes.
 - This suite works out of the box with the canonical CIWA+ dataset. You shouldn't have to reorganize any files. If you still want to use the old dataset layout from the other repos, I plan to add support for that eventually.
 - No need for per-dataset files containing classes. It was the same file in every dataset, so the class specifications now live at the top of dataset.py. I can add support for a per-dataset `.csv` file later, which would override any hard-coded class specifications.
 - Defaults are set to match earlier papers / repos where applicable.
 - Every run is accompanied by a `settings.json` file, to aid in reproducibility during future runs

*\* Subjective*
