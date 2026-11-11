# Segment-nouveau

**nouveau**  
*adjective*  
nou·veau - /nü-ˈvō/  
Definition: newly arrived or developed

A (hopefully more) modular framework for the implementation, training, and testing of various Semantic Segmentation models, rebuilt from the ground up for CIWA+ purposes. Should also be easy enough to modify for other purposes if you know what you're doing and aren't using this as part of the CIWA+ project. 

For reference: This repo was built based on [AIISLab segment-main](https://github.com/AIISLab/segment-main) repo, which is, as far as I can tell, the successor to [George Seif's Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite) repo that appears to have been used in the earlier CIWA+ paper.

### Important notes:
 - Runs on an NVIDIA GPU by default (`device=cuda`). `device=cpu` works but is much slower.
 - Console logging only shows the time, .log files show the entire timestamp. Preferences for this functionality live in `conf/hydra/job_logging/custom.yaml`
 - Each crop has its own `train_crop` value in `conf/crop/<crop>.yaml` which will need to be set for any crops added down the line.
 - This project uses [hydra](https://hydra.cc/) now, rather than argparse. This changes the way command line arguments are input (`arg=val` instead of `--arg val`) and adds some new functionalities.
 - When passing filepaths for the data_root parameter, use full paths; $HOME/... works, a bare ~/... doesn't (%USERPROFILE%\\... in Windows CMD rather than $HOME/...).
 - Scripts that work on one model take checkpoint (a run folder or the .pt inside it). Scripts that can work on many runs take folder (a single run, a crop folder, or a whole sweep).

## README contents:

[Dataset Requirements](#dataset-requirements)

[Getting Started](#getting-started)

[Training a Model](#training-a-model)

[Evaluating a Model](#evaluating-a-model)

[Summarizing Results](#summarizing-results)

[Generating Figures](#generating-figures)

[Predicting on New Images](#predicting-on-new-images)

[Extending the Suite](#extending-the-suite)

[Repo Contents](#repo-contents)

[Design Decisions](#design-decisions-or-alternatively-why-would-you-rebuild-a-perfectly-good-tool)

## Dataset Requirements:  
Your dataset should follow the example structure below. In short: One directory containing directories for each crop, each with their own test, train, and valid folders, with each split having a visual and label folder. Extra folders are ignored.

Visual images are matched with their labels via their stems, where the labels have "_L" appended to their stem (i.e. the label for visual/img.jpg should be at label/img_L.png). Visual images can be `.jpg`, `.jpeg`, `.png`, or `.tif`/`.tiff` and any `-Visual` ending or leftover extensions are stripped before matching. For example, IMG1.JPG-Visual.jpeg pairs with IMG1_L.png.

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

**Available models**: 
- `deeplabv3` (DeepLabV3 with a MobileNet backbone, ~11.0M parameters)
- `frrn_a` (full FRRN-A, ~17.7M parameters, needs a lot of GPU memory)
- `frrn_a_lite` (reduced FRRN-A, ~1.4M parameters, width set by `model.base`)
- `fcdensenet` (FC-DenseNet103 matching the ~25.7M parameter version used in the 2026 CIWA tomato paper, also very memory hungry)

See [Checking a model's size and memory](#checking-a-models-size-and-memory) before picking a `batch_size`.

**Example single run**:

```bash
python train.py data_root=/full/path/to/dataset crop=tomato epochs=100 patience=7 lr=1e-4 seed=0
```

Checkpoints are saved to `checkpoints/<crop>/<timestamp>/<model>-best.pt` by default, along with a `.hydra/` folder that contains `config.yaml` and `overrides.yaml` detailing how the training was configured, and the `history.json` and `train.log` that accompanied their training. A `git.json` file is written in the same location which details the state of the repo at the time the training was run. Models are automatically evaluated against each split in `eval_splits` (test split by default) corresponding to the crop they were trained on, and evaluation results are in `eval/<crop>-<split>/metrics.json`.

**Example multi-run**:

```bash
python train.py --multirun data_root=/full/path/to/dataset crop=tomato epochs=100 patience=7 lr=1e-4,1e-3 seed=0,1,2
```

Sweeps (see above multi-run example) output to `checkpoints/sweeps/<timestamp>/<crop>/<settings>/`, where `<settings>` refers to the unique settings used to generate that specific checkpoint, e.g., `epochs-100+lr-1e-4+patience-7+seed-0`. Non-experimental keys like `device` or `num_workers` are left out of these settings, so changing them won't split a sweep's groupings. 

*Note: When passing a path, use `$HOME` instead of `~`, and if passing a list, arguments must be quoted (see `eval_splits` and `crop.train_crop` below and note the single quotes around the flags)*

*Windows CMD: use `%USERPROFILE%` instead of `$HOME`, and double quotes instead of single quotes around list arguments, e.g. `"eval_splits=[valid,test]"`. PowerShell works with the commands as written.*

### Command line arguments (Training)

| Argument           | Description                                                                                                                                                     | Example                            |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `data_root`        | Root dataset directory, should have a subdirectory for the crop you want to train on - Required                                                                 | `data_root=/home/ciwa-full-split/` |
| `crop`             | The crop (name of subdirectory within the dataset root directory) you want to train on - Required                                                               | `crop=tomato`                      |
| `model`            | Which model to train                                                                                                                                            | `model=deeplabv3`                  |
| `optimizer`        | Which optimizer to use during training                                                                                                                          | `optimizer=rmsprop`                |
| `loss`             | Specifies the loss function to be used during training                                                                                                          | `loss=ce_dice`                     |
| `loss.dice_weight` | Determines dice weight to be used for loss - cross-entropy plus dice weight multiplied by dice loss (only to be used if using `loss=ce_dice`)                   | `loss.dice_weight=1.0`             |
| `epochs`           | Maximum number of epochs to train for (passes over the training set)                                                                                            | `epochs=100`                       |
| `monitor`          | Which metric to monitor during training for early stopping and determining the best model to save                                                               | `monitor=valid_loss`               |
| `patience`         | How many epochs to continue training for without seeing improvement in the monitored value                                                                      | `patience=7`                       |
| `batch_size`       | Number of images in each training batch                                                                                                                         | `batch_size=8`                     |
| `lr`               | The learning rate of the model                                                                                                                                  | `lr=1e-4`**                        |
| `crop.train_crop`  | The size of the random crop that is applied to training images (h x w)                                                                                          | `'crop.train_crop=[256,480]'`      |
| `output_dir`       | The folder this script should output to                                                                                                                         | `output_dir=checkpoints`           |
| `seed`             | Number to use as a random seed for anything that can be seeded                                                                                                  | `seed=42`                          |
| `eval_splits`      | Which splits the model should be automatically evaluated against after training is completed                                                                    | `'eval_splits=[test, valid]'`*     |
| `sizing`           | How images are sized for the model: native = random training crops, full size imgs for validation/scoring; stretch512 resizes everything to 512x512             | `sizing=stretch512`                |
| `device`           | Device where everything should happen: `cuda`, `cuda:<n>` for a specific GPU, or `cpu`.                                                                         | `device=cuda:1`                    |
| `num_workers`      | Number of processes loading data in parallel - note that this does in fact result in different random crops, so keep it the same when trying to reproduce a run | `num_workers=8`                    |

*`'eval_splits=[]'` disables automatic post-training evaluation

**`lr=1e-4` and `lr=0.0001` will create different folders despite being equivalent values.

### Choosing a GPU for training

On a machine containing multiple GPUs (e.g. a shared lab PC), check which ones are free before beginning training:

```bash
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.used,memory.total,utilization.gpu --format=csv
```

In torch, GPUs are numbered fastest-first. This might not necessarily match the order in `nvidia-smi`. 

You can set `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the terminal window you plan on training from so that the two numbering systems match:

- bash: `export CUDA_DEVICE_ORDER=PCI_BUS_ID`
- fish: `set -x CUDA_DEVICE_ORDER PCI_BUS_ID`
- ps: `$env:CUDA_DEVICE_ORDER = "PCI_BUS_ID"`
- cmd: `set CUDA_DEVICE_ORDER=PCI_BUS_ID`

This setting lasts until that terminal is exited. The first lines of `train.log` note the GPU in use, its bus ID, and its free memory (and will warn you if that's less than half the total memory). If you're on Windows trust `nvidia-smi` rather than Task Manager for GPU usage information. A run on a GPU without enough free memory may sit at 0% instead of crashing.

### Checking a model's size and memory

`tools/model_info.py` builds a model, counts its parameters, checks that an odd-sized image comes back at the same size, and measures peak GPU memory for one training step:


**Example check**:

```bash
python -m tools.model_info model=frrn_a_lite batch_size=4
```

| Argument     | Description                                                               | Example                           |
| ------------ | ------------------------------------------------------------------------- | --------------------------------- |
| `model`      | Which model to check, plus any of its options                             | `model=frrn_a_lite model.base=32` |
| `size`       | Height, width of the test batch (defaults to the big crops' `train_crop`) | `'size=[256,480]'`                |
| `batch_size` | Batch size for the memory check                                           | `batch_size=4`                    |
| `device`     | GPU to measure on (`cpu` skips the memory check)                          | `device=cuda:1`                   |

 If the batch doesn't fit, it logs an out-of-memory warning instead of crashing. Try a smaller batch_size. 
 
 Peak memory only counts this process, so leave some headroom on a shared GPU. Logs go to `outputs/model_info/<model>/<timestamp>/`.

## Evaluating a model

Models are scored using the sizing they were trained with, and always against the original labels at full size. `metrics.json` records sizing and code version used.

**Example single evaluation**:

```bash
python evaluate.py folder=checkpoints/<crop>/<timestamp>/ crop=tomato split=test
```

Evaluation results are saved to `<run>/eval/<crop>-<split>/metrics.json`, where "crop" represents the evaluated dataset and "split" the evaluated split, alongside the accompanying `evaluate.log`.

**Example multi-evaluation**:

```bash
python evaluate.py folder=checkpoints/sweeps/<timestamp>/ crop=tomato
```

Evaluation of a whole sweep will score any `*-best.pt` files contained in the `folder` or any of `folder`'s subdirectories. An overall evaluate.log will be created in `<folder>/eval/<crop>-<split>/` which will list each run along with any failures. One failed run should not stop the others.

*Note: you can score a model on a crop different from the one it trained on, since `crop` is just the dataset being scored*

### Command line arguments (Evaluation)

| Argument      | Description                                                                                                                             | Example                                          |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `folder`      | A run folder, any folder containing runs, or the .pt file corresponding to a run - Required                                             | `folder=checkpoints/tomato/2026-10-11_07-48-57/` |
| `crop`        | The crop (name of subdirectory within the dataset root directory) you want to use in evaluation - Required                              | `crop=tomato`                                    |
| `split`       | The split the model should consider for evaluation                                                                                      | `split=test`                                     |
| `data_root`   | Root dataset directory override. Use when the data_root in the corresponding `.hydra/config.yaml` does not exist on the current machine | `data_root=/home/ciwa-full-split/`*              |
| `device`      | Device where everything should happen: `cuda`, `cuda:<n>` for a specific GPU, or `cpu`.                                                 | `device=cuda:1`                                  |
| `num_workers` | Number of processes loading data in parallel                                                                                            | `num_workers=8`                                  |

*`null` as a value for this option means "use the one the model used during training"

## Summarizing results

**Example summarization**:

```bash
python summarize.py folder=checkpoints/sweeps/<timestamp>/
```

Takes a set of scored runs (as a result of automatic evaluation from `train.py` or manually via `evaluate.py`), reads the corresponding `metrics.json` for each run, groups runs that differ in seed only, and reports the mean ± standard deviation across stats for those runs. Prints one table per scored crop + split. Writes output to `<folder>/summary/summary.json` and `<folder>/summary/summarize.log`. The log holds the same table that's printed to the console.

| Argument  | Description                                                                   | Example                                         |
| --------- | ----------------------------------------------------------------------------- | ----------------------------------------------- |
| `folder`  | A sweep, crop folder inside a sweep, or any folder containing runs - Required | `folder=checkpoints/sweeps/2026-10-22_18-02-44` |
| `metrics` | Which scores to show, default is all 8                                        | `'metrics=[sunlit_iou,f1]'`                     |

## Generating figures

**Example generation for a sweep**:

```bash
python figures.py folder=checkpoints/sweeps/<timestamp>/ metric=precision
```

**Example generation for a single run**:

```bash
python figures.py folder=checkpoints/<crop>/<timestamp>/ metric=precision
```

To generate figures, the script reads `history.json` from the model's training to generate `curves.png` plotting metrics against epochs. It also reads `eval/*/metrics.json` from the model evaluation to generate confusion matrices, one per scored split, output as `confusion-<crop>-<split>.png`. Finally it reads `summary/summary.json` from the summarization script to output one `comparison-<crop>-<split>.png` per section. The script also writes `overview-<split>.png` per scored split made from the same summary which shows all crops side by side, one bar per setting, with mean ± std across seeds. The overview only considers models **scored on the crop they were trained on** and is skipped with a log message when the summary covers fewer than two crops (since that would just repeat that single crop's comparison chart). If `summary.json` is missing, it will log a message telling you to run `summarize.py` first. Runs without a `history.json` won't be found (so won't have confusion matrices generated), but missing files in general will not result in an error.

Everything output by `figures.py` goes in `<folder>/figures/`, where per-run figures mirror the run folder layout with `figures.log` next to them. Re-running the script overwrites existing figures, so they will reflect the last `metric` used when running it.

*Note: The dashed line on the curve marks the saved (best according to `monitor` value used in training) epoch. It won't necessarily be at the peak of the `metric` being plotted, unless it matches the `monitor` value. Also, confusion matrices show percentages as a share of all pixels in the split with a fixed color scale so shades are comparable across figures. The titles for confusion matrices will always show `sunlit IoU` regardless of `metric`*

| Argument | Description                                                                                                                                                                                                | Example                                         |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `folder` | A sweep, crop folder inside a sweep, or any folder containing runs - Required                                                                                                                              | `folder=checkpoints/sweeps/2026-10-22_18-02-44` |
| `metric` | Which metric to use for the curves, comparison, and overview charts - Default is `sunlit_iou`, must be one of [`accuracy`, `precision`, `recall`, `fpr`, `f1`, `sunlit_iou`, `non_sunlit_iou`, `mean_iou`] | `metric=f1`                                     |

## Predicting on new images

**Example prediction (folder)**:

```bash
python predict.py checkpoint=checkpoints/<crop>/<timestamp>/ input=<path/to/input/images> save_logits=True save_overlays=False
```

**Example prediction (single image)**:

```bash
python predict.py checkpoint=checkpoints/<crop>/<timestamp>/ input=<path/to/input/image.png> save_logits=False save_overlays=True
```

Outputs are saved to `<run>/predict/<timestamp>` with a `predict.log` file and organized into folders corresponding to what they are: `masks/`, `logits/`, `overlays/`. Masks use the label colors and are named `<stem>_pred.png` to line up with `<stem>_L.png`. The other two outputs follow a similar scheme, with `_pred.png` being replaced by `_logits.npy` and `_overlay.png`. The overlays themselves tint only pixels predicted as `sunlit`. Saving the optional outputs (logits and overlays) can result in many large files being saved to your disk (~12 MB for logits per almond/citrus/grape image in the CIWA+ dataset), so saving them defaults to False.

*Note: Logits are saved as [2, H, W] float32, channel 0 non-sunlit, channel 1 sunlit - Outputs for models trained with stretch sizing are resized back to native size*

| Argument        | Description                                                                                 | Example                                                               |
| --------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `checkpoint`    | A run folder or the .pt file inside of it - Required                                        | `checkpoint=checkpoints/tomato/2026-10-22_18-02-44/deeplabv3-best.pt` |
| `input`         | Either a single image or a folder containing images to generate predictions from - Required | `input=$HOME/dataset/tomato/test/visual`                              |
| `save_logits`   | Whether logits should be saved with the predicted masks                                     | `save_logits=True`                                                    |
| `save_overlays` | Whether overlays should be saved with the predicted masks                                   | `save_overlays=False`                                                 |
| `device`        | Device where everything should happen: `cuda`, `cuda:<n>` for a specific GPU, or `cpu`.     | `device=cuda:1`                                                       |

## Extending the suite

### Adding a new model

*See `models/__init__.py`*

New models require adding a config file under `conf/model/<name>.yaml`, containing `name: <name>` where `<name>` is its key in `MODELS`. Any other keys in that file get passed to the model's constructor and can be overridden in the command line (e.g. `model.base=32` for `frrn_a_lite`). Run the following command to check the new model's parameter count, output shape and memory use before training:

```bash
python -m tools.model_info model=<name>
```

*Note: You only need to do the following if the model was trained with `sizing=native`. `check_sizing.py` only applies to models trained with native sizing, it stops with a warning otherwise. This is by design.*

After adding and training a new model, run `python -m tools.check_sizing checkpoint=<run> crop=<crop>` for each crop you plan to report. Command line arguments for `check_sizing.py`, `checkpoint` and `crop`, are both required. This should tell you whether or not the model's performance is being harmed by an incorrect `size_multiple`. The first two values printed by `check_sizing.py` should be relatively close to each other. If not, you'll want to increase the model's size multiple, e.g. 8 -> 16, 16 -> 32, and run `check_sizing.py` again. Existing checkpoints should pick up the change without retraining, though you should retrain before reporting results since validation used the old padding during training. If the third value is exceptionally low, the model is breaking down when it sees black-padded images. This should be a non-issue for now since nowhere in this repo uses black padding.

### Adding a new crop

Create `conf/crop/<crop>.yaml` with `name` and `train_crop`, following `tomato.yaml`. The folder name under your `data_root` must match the `name` of the corresponding `crop.yaml`. `train_crop` must fit inside the smallest image in the dataset, and ideally be divisible by the model's `size_multiple` (32 for FCDenseNet, 16 for DeepLabV3 and FRRN-A, 4 for FRRN-A lite).

### Adding a new loss function

1. Write the loss as a function or class in `components/losses.py`
2. Add it to the LOSSES registry (also in `components/losses.py`)
3. Create `conf/loss/<name>.yaml`, with `name: <name>`, plus any arguments it takes (see `conf/loss/ce_dice.yaml` for example)

### Adding a new sizing method

1. Create `conf/sizing/<name>.yaml`
2. add a branch in `build_training_transform` (`components/transforms.py`)
3. add a branch in `predict_logits` (`components/inference.py`)

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
| `figures.py`   | Figure generation for scored and summarized runs                                                           |
| `predict.py`   | Runs a specified model to generate predicted masks for given image(s)                                      |
| `summarize.py` | Summarizes metrics for a given directory containing one or more metrics.json file(s) anywhere inside of it |
| `train.py`     | Training logic and loop, this is what you run to train the models                                          |

## Design decisions, or alternatively "Why would you rebuild a perfectly good* tool?"

 - Native sizing of images is the default. Training uses random crops so the models aren't learning from distorted images. `sizing=stretch512` reproduces the 2026 tomato paper's stretch-to-512x512 for comparison sake. In any case, scores are always computed against the original label at full size.
 - When a model is run, images are padded out to clean multiples of a size specified by the model implementation. This padding uses mirroring rather than black pixels. Never pad with black. With the default training crop behavior, this only affects validation and testing
 - Validation images handled with a batch size of 1. This is to allow for the differently sized images within the tomato dataset. Previous implementations stretched everything to 512x512, so never had batches of mixed sizes.
 - This suite works out of the box with the canonical CIWA+ dataset. You shouldn't have to reorganize any files. If you still want to use the old dataset layout from the other repos, I plan to add support for that eventually.
 - No need for per-dataset files containing classes. It was the same file in every dataset, so the class specifications now live at the top of `components/dataset.py`. I can add support for a per-dataset `.csv` file later, which would override any hard-coded class specifications.
 - Defaults are set to match earlier papers / repos where applicable.
 - Every run is accompanied by a `.hydra/config.yaml` file, to aid in reproducibility during future runs
 - Every run records the git commit and whether or not the repo was dirty at the time (`git.json` and `metrics.json` for scoring)

*\* Subjective*

[Back to Top](#segment-nouveau)
