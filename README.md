# Segment-nouveau

**nouveau**  
*adjective*  
nou·veau - /nü-ˈvō/  
Definition: newly arrived or developed

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