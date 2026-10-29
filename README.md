# Segment-nouveau

**nouveau**  
*adjective*  
nou·veau - /nü-ˈvō/  
Definition: newly arrived or developed

A (hopefully more) modular framework for the implementation, training, and testing of various Semantic Segmentation models, rebuilt from the ground up for CIWA+ purposes. Should also be easy enough to modify for other purposes if you know what you're doing and aren't using this as part of the CIWA+ project. 

For reference: This repo was built based on [AIISLab segment-main](https://github.com/AIISLab/segment-main) repo, which is, as far as I can tell, the successor to [George Seif's Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite) repo that appears to have been used in the earlier CIWA+ paper.

### Important notes:
 - Running these scripts requires an NVIDIA GPU
 - Each crop has its own `train_crop` value in `conf/crop/<crop>.yaml`
 - This project uses [hydra](https://hydra.cc/) now, rather than argparse. This changes the way command line arguments are input (`arg=val` instead of `--arg val`) and adds some new functionalities.
 - When passing filepaths for the data_root parameter, use full paths; $HOME/... works, a bare ~/... doesn't.

## Dataset Requirements:  
Your dataset should follow the example structure below. In short: One directory containing directories for each crop, each with their own test, train, and valid folders, with each split having a visual and label folder. Extra folders are ignored.

Visual images are matched with their labels via their stems, where the labels have "_L" appended to their stem (i.e. the label for visual/img.jpg should be at label/img_L.png). Visual images can be `.jpg` or `.jpeg`, and any `-Visual` ending or leftover `.jpg`/`.JPG` are stripped before matching. For example, IMG1.JPG-Visual.jpeg pairs with IMG1_L.png.

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

**Example single run**: `python train.py data_root=path/to/dataset crop=tomato epochs=100 patience=7 lr=1e-4 seed=0`

Checkpoints are saved to `checkpoints/<crop>/<timestamp>/<model>-best.pt` by default, along with a `.hydra/` folder that contains `config.yaml` and `overrides.yaml` detailing how the training was configured, and the `history.json` and `train.log` that accompanied their training. Models are automatically evaluated against each split in `eval_splits` (test split by default) corresponding to the crop they were trained on, and evaluation results are in `eval/<crop>-<split>/metrics.json`.

**Example multi-run**: `python train.py --multirun data_root=path/to/dataset crop=tomato epochs=100 patience=7 lr=1e-4,1e-3 seed=0,1,2`

Sweeps (see above multi-run example) output to `checkpoints/sweeps/<timestamp>/<crop>/<settings>/`, where `<settings>` refers to the unique settings used to generate that specific checkpoint, e.g., `epochs-100+lr-0.0001+patience-7+seed-0`

*Note: When passing a path, use `$HOME` instead of `~`, and if passing a list, arguments must be quoted (see `eval_splits` and `crop.train_crop` below and note the single quotes around the flags)*

### Command line arguments (Training)

| Argument          | Description                                                                                                                  | Example                            |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `data_root`       | Root dataset directory, should have a subdirectory for the crop you want to train on - Required                              | `data_root=/home/ciwa-full-split/` |
| `crop`            | The crop (name of subdirectory within the dataset root directory) you want to train on - Required                            | `crop=tomato`                      |
| `model`           | Which model to train                                                                                                         | `model=deeplabv3`                  |
| `optimizer`       | Which optimizer to use during training                                                                                       | `optimizer=rmsprop`                |
| `dice_weight`     | Determines dice weight to be used for loss - cross-entropy plus dice weight multiplied by dice loss - 0 to disable dice loss | `dice_weight=1.0`                  |
| `epochs`          | Maximum number of epochs to train for (passes over the training set)                                                         | `epochs=100`                       |
| `monitor`         | Which metric to monitor during training for early stopping and determining the best model to save                            | `monitor=valid_loss`               |
| `patience`        | How many epochs to continue training for without seeing improvement in the monitored value                                   | `patience=7`                       |
| `batch_size`      | Number of images in each training batch                                                                                      | `batch_size=8`                     |
| `lr`              | The learning rate of the model                                                                                               | `lr=1e-4`                          |
| `crop.train_crop` | The size of the random crop that is applied to training images (h x w)                                                       | `'crop.train_crop=[256,480]'`      |
| `output_dir`      | The folder this script should output to                                                                                      | `output_dir=checkpoints`           |
| `seed`            | Number to use as a random seed for anything that can be seeded                                                               | `seed=42`                          |
| `eval_splits`     | Which splits the model should be automatically evaluated against after training is completed                                 | `'eval_splits=[test, valid]'`*     |

*`'eval_splits=[]'` disables automatic post-training evaluation

## Evaluating a model

**Example single evaluation**: `python evaluate.py folder=checkpoints/<crop>/<timestamp>/ crop=tomato split=test`

Evaluation results are saved to `<run>/eval/<crop>-<split>/metrics.json`, where "crop" represents the evaluated dataset and "split" the evaluated split, alongside the accompanying `evaluate.log`.

**Example multi-evaluation**: `python evaluate.py folder=checkpoints/sweeps/<timestamp>/ crop=tomato`

Evaluation of a whole sweep will score any `*-best.pt` files contained in the `folder` or any of `folder`'s subdirectories. An overall evaluate.log will be created in `<folder>/eval/<crop>-<split>/` which will list each run along with any failures. One failed run should not stop the others.

*Note: you can score a model on a crop different from the one it trained on, since `crop` is just the dataset being scored*

### Command line arguments (Evaluation)

| Argument    | Description                                                                                                                             | Example                                          |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `folder`    | A run folder, or any folder containing runs - Required                                                                                  | `folder=checkpoints/tomato/2026-10-11_07-48-57/` |
| `crop`      | The crop (name of subdirectory within the dataset root directory) you want to use in evaluation - Required                              | `crop=tomato`                                    |
| `split`     | The split the model should consider for evaluation                                                                                      | `split=test`                                     |
| `data_root` | Root dataset directory override. Use when the data_root in the corresponding `.hydra/config.yaml` does not exist on the current machine | `data_root=/home/ciwa-full-split/`*              |

*`null` as a value for this option means "use the one the model used during training"

## Summarizing results

**Example summarization**: `python summarize.py folder=checkpoints/sweeps/<timestamp>/`

Takes a set of scored runs (as a result of automatic evaluation from `train.py` or manually via `evaluate.py`), reads the corresponding `metrics.json` for each run, groups runs that differ in seed only, and reports the mean ± standard deviation across stats for those runs. Prints one table per scored crop + split. Writes output to `<folder>/summary/summary.json` and `<folder>/summary/summarize.log`. The log holds the same table that's printed to the console.

| Argument  | Description                                                                   | Example                                         |
| --------- | ----------------------------------------------------------------------------- | ----------------------------------------------- |
| `folder`  | A sweep, crop folder inside a sweep, or any folder containing runs - Required | `folder=checkpoints/sweeps/2026-10-22_18-02-44` |
| `metrics` | Which scores to show, default is all 8                                        | `'metrics=[sunlit_iou,f1]'`                     |

## Extending the suite

### Adding a new model

*See `models/__init__.py`*

New models require adding a config file under `conf/model/<name>.yaml`, containing `name: <name>` where `<name>` is its key in `MODELS`.

After adding and training a new model, run `python -m tools.check_sizing checkpoint=<run folder> crop=<crop>` for each crop you plan to report. Command line arguments for `check_sizing.py`, `checkpoint` and `crop`, are both required.

### Adding a new crop

Create `conf/crop/<crop>.yaml` with `name` and `train_crop`, following `tomato.yaml`. The folder name under your `data_root` must match the `name` of the corresponding `crop.yaml`. `train_crop` must fit inside the smallest image in the dataset, and ideally be divisible by the model's `size_multiple` (16 for DeepLabV3)

### Adding a new loss function

*See `components/losses.py`*

### Adding a new optimizer

*See `components/optimizers.py`*

### Adding new transforms

*See `components/transforms.py`*

## Repo Contents
| File           | Contents                                                                                                   |
| -------------- | ---------------------------------------------------------------------------------------------------------- |
| `components/`  | Losses, metrics, optimizers, transforms, dataset definition, inference and scoring logic                   |
| `conf/`        | Hydra configs                                                                                              |
| `models/`      | Model wrappers and padding helper                                                                          |
| `tools/`       | One-off checks                                                                                             |
| `evaluate.py`  | Evaluation logic and loop, this is what you run to evaluate the models                                     |
| `summarize.py` | Summarizes metrics for a given directory containing one or more metrics.json file(s) anywhere inside of it |
| `train.py`     | Training logic and loop, this is what you run to train the models                                          |

## Design decisions, or alternatively "Why would you rebuild a perfectly good* tool?"

 - Images are never stretched. If models expect inputs to have a specific aspect ratio, we instead take a randomly cropped window from our full images, so that the model isn't learning from a distorted view of our data. This follows the approach of the suite used in the earlier papers.
 - When a model is run, images are padded out to clean multiples of a size specified by the model implementation. This padding uses mirroring rather than black pixels. Never pad with black. With the default training crop behavior, this only affects validation and testing
 - Validation images handled with a batch size of 1. This is to allow for the differently sized images within the tomato dataset. Previous implementations stretched everything to 512x512, so never had batches of mixed sizes.
 - This suite works out of the box with the canonical CIWA+ dataset. You shouldn't have to reorganize any files. If you still want to use the old dataset layout from the other repos, I plan to add support for that eventually.
 - No need for per-dataset files containing classes. It was the same file in every dataset, so the class specifications now live at the top of `components/dataset.py`. I can add support for a per-dataset `.csv` file later, which would override any hard-coded class specifications.
 - Defaults are set to match earlier papers / repos where applicable.
 - Every run is accompanied by a `.hydra/config.yaml` file, to aid in reproducibility during future runs

*\* Subjective*
