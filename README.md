# Segment-nouveau

**nouveau**  
*adjective*  
nou·veau - /nü-ˈvō/  
Definition: newly arrived or developed

A (hopefully more) modular framework for the implementation, training, and testing of various Semantic Segmentation models, rebuilt from the ground up for CIWA+ purposes. Should also be easy enough to modify for other purposes if you know what you're doing and aren't using this as part of the CIWA+ project. 

For reference: This repo was built based on [AIISLab segment-main](https://github.com/AIISLab/segment-main) repo, which is, as far as I can tell, the successor to [George Seif's Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite) repo that appears to have been used in the earlier CIWA+ paper.

### Important notes:
 - Requires NVIDIA GPU
 - Default crop and pad sizes chosen for the tomato dataset, other datasets will need this set to be specific to them.

## Dataset Requirements:  
Your dataset should follow the example structure below. In short: One directory containing directories for each crop, each with their own test, train, and valid folders, with each split having a visual and label folder. Extra folders are ignored.

Visual images are matched with their labels via their stems, where the labels have "_L" appended to their stem (i.e. the label for visual/img.jpg should be at label/img_L.png)

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

*Project is built and tested with python 3.11*

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

*Must be done in this order, some packages may depend on torch. Also note that Torch is pinned to a specific CUDA build - 12.6*

```bash
pip install -r requirements-torch.txt
pip install -r requirements.txt
```

## Training a model

*Coming Soon*

Checkpoints are saved to `checkpoints/crop/timestamp/architecture-best.pt` by default, along with the `settings.json` that was used to produce them.

## Evaluating a model

*Coming Soon*

## Extending the suite

### Adding a new model

*See `models/__init__.py`*

### Adding a new loss function

*See `losses.py`*

### Adding a new optimizer

*See `optimizers.py`*

### Adding new transforms

*See `transforms.py`*

## Repo Contents
| File                     | Contents                                                                                |
|--------------------------|-----------------------------------------------------------------------------------------|
| `models/`                | Builder and implementation files for all implemented models                             |
| `dataset.py`             | Finds images and their corresponding labels, loads them, ensures they match             |
| `losses.py`              | Builder and implementations for the loss functions used during model training           |
| `optimizers.py`          | Builder and implementations for the optimizers used during model training               |
| `README.md`              | What you're reading now.                                                                |
| `requirements.txt`       | Non-torch project requirements                                                          |
| `requirements-torch.txt` | Torch-specific project requirements                                                     |
| `train.py`               | Training logic and loop, this is what you run to train the models                       |
| `transforms.py`          | Builder and implementations for any transforms used during model training or evaluation |



## Design decisions, or alternatively "Why would you rebuild a perfectly good* tool?"

 - Images are never stretched. If models expect inputs to have a specific aspect ratio, we instead take a randomly cropped window from our full images, so that the model isn't learning from a distorted view of our data. This follows the approach of the suite used in the earlier papers.
 - Validation images are padded, and padded pixels are ignored. This is used to fix an issue with inconsistent image sizes within batches that was silently ignored in the most recent CIWA+ paper, due to the aforementioned stretching of images.
 - This suite works out of the box with the canonical CIWA+ dataset. You shouldn't have to reorganize any files. If you still want to use the old dataset layout from the other repos, I plan to add support for that eventually.
 - No need for per-dataset files containing classes. It was the same file in every dataset, so the class specifications now live at the top of dataset.py. I can add support for a per-dataset `.csv` file later, which would override any hard-coded class specifications.
 - Defaults are set to match earlier papers / repos where applicable.
 - Every run is accompanied by a `settings.json` file, to aid in reproducibility during future runs

*\* Subjective*
