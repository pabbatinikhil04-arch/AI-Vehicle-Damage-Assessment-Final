# AI Vehicle Damage Assessment

A deep-learning-based vehicle damage assessment project that combines image authenticity screening, damage detection, damage classification, damage segmentation, damage-area estimation, severity assessment, and project-defined insurance eligibility prediction.

This repository contains project versions, dataset-related resources, model files, outputs, and supporting materials for the Deep Learning end-semester project.

## Project Overview

The application accepts a vehicle image and processes it through a modular assessment pipeline:

1. **Image authenticity screening** — predicts whether an image is `REAL`, `AI_GENERATED`, or `UNCERTAIN`. Images flagged as AI-generated are stopped from proceeding to damage assessment; uncertain images may require a clearer or separately reviewed image.
2. **Photo quality checks** — provides heuristic feedback about whether the image is suitable for analysis.
3. **Binary damage detection** — checks whether damage is present before detailed classification.
4. **Eight-class damage classification** — predicts the most likely damage category.
5. **Damage segmentation** — identifies the predicted damaged pixels in the image.
6. **Damage-area estimation** — calculates the percentage of the image marked as damaged.
7. **Severity assessment and scoring** — converts the estimated area and classification confidence into project-defined assessment categories.
8. **Results display** — presents the outputs through a web application.

The project is an academic prototype. Its eligibility result is based on project-defined rules and is **not an actual insurance underwriting decision**, repair-cost estimate, or guarantee of claim approval. Image-authenticity predictions are probabilistic and should not be treated as forensic proof.

## Damage Categories

The eight-class classifier uses these categories:

- Bumper Dent
- Bumper Scratch
- Door Dent
- Door Scratch
- Glass Shatter
- Head Lamp
- Tail Lamp
- Unknown

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React |
| Backend/API | Python, FastAPI |
| Damage classification | EfficientNet-B0 and ResNet18 (comparison versions) |
| Binary damage detection | EfficientNet-B0 |
| Damage segmentation | U-Net with a ResNet18 encoder |
| Image processing and inference | PyTorch-based deep-learning pipeline |
| Version control | Git and Git LFS for large files |

## System Workflow

```text
Vehicle image upload
        |
        v
Image preprocessing and authenticity screening
        |
        +---- AI_GENERATED / UNCERTAIN ---> Flag or stop for review
        |
        v
Binary damage detection
        |
        v
Eight-class damage classification
        |
        v
U-Net damage segmentation
        |
        v
Damaged-area percentage estimation
        |
        v
Severity level + assessment score
        |
        v
Project-defined eligibility category
        |
        v
Results shown in the React interface
```

## Dataset Information

### Damage classification dataset

The prepared classification dataset contains **1,512 unique images** across eight classes. The data were split using stratified sampling:

| Split | Images |
|---|---:|
| Training | 1,058 |
| Validation | 227 |
| Test | 227 |
| **Total** | **1,512** |

The report also documents an AI-generated-image collection containing **6,226 images** associated with DALL·E, Kandinsky, Midjourney, SDXL, and UnCLIP. The report notes that one short training run showed evidence of a preprocessing-related shortcut, so the reported authenticity-detector metrics should not be interpreted as proof of generalization.

Dataset and output folders visible in the repository include `AI_VEHICLE_DATASET`, `outputs`, and other project/version directories. Check the relevant folder and dataset terms before running a particular version; paths and files can differ between versions.

## Reported Results

The following values are reported in the project report. They describe the documented experiments and should not be assumed to represent every version or every image encountered in deployment.

### Eight-class damage classification

Both classifiers were evaluated on the same held-out test set of 227 images.

| Metric | EfficientNet-B0 | ResNet18 |
|---|---:|---:|
| Accuracy | 85.02% | **88.11%** |
| Weighted precision | 85.94% | **89.02%** |
| Weighted recall | 85.02% | **88.11%** |
| Weighted F1-score | 84.95% | **87.78%** |

ResNet18 performed better on this test split and was selected as the preferred eight-class classifier in the report. This result applies to the evaluated dataset and does not establish that ResNet18 is universally better.

### Damage segmentation and area estimation

| Metric | Reported result |
|---|---:|
| Segmentation model | U-Net with ResNet18 encoder |
| Dice score | 81.01% |
| Intersection over Union (IoU) | 73.81% |
| Segmentation threshold | 0.60 |
| Damage-area MAE | 1.2871 percentage points |
| Damage-area RMSE | 3.5767 percentage points |
| Damage-area evaluation samples | 123 |
| Correlation between predicted and reference area | 0.8999 |

The damage-area values represent the share of image pixels predicted as damaged. They do not directly measure physical damage depth, structural integrity, repair cost, or claim value.

### Image authenticity evaluation

The report lists two separate evaluation runs. The report does not establish that both runs correspond to the same deployed application version, so they are kept separate here.

| Metric | Evaluation Run A | Evaluation Run B |
|---|---:|---:|
| Samples | 1,162 | 263 |
| Accuracy | 99.91% | 98.48% |
| Precision | 99.89% | 98.64% |
| Recall | 100.00% | 99.54% |
| F1-score | 99.95% | 99.09% |
| ROC-AUC | 1.000 | 0.999 |

These high scores are specific to the reported evaluation sets. The report cautions that they do not independently establish generalization or forensic authenticity.

## Project-Defined Severity and Assessment Rules

### Damage severity

Severity is assigned from the predicted damaged-area percentage:

| Estimated damaged area | Severity |
|---|---|
| Less than 5% | Low |
| 5% to less than 15% | Moderate |
| 15% to less than 30% | High |
| 30% or above | Very High |

### Assessment score

The project computes the score using:

```text
Assessment Score = (0.60 × Damage Area %) + (0.40 × Classification Confidence %)
```

The inputs are expressed on a 0–100 scale, and the resulting score is restricted to that range.

| Score | Project-defined assessment category |
|---|---|
| Below 25 | Low Assessment |
| 25 to less than 50 | Moderate Assessment |
| 50 to less than 75 | High Assessment |
| 75 or above | Very High Assessment |

For the project demonstration, a score of **25 or above** is labelled **Insurance Eligible**, while a score below 25 is labelled **Not Insurance Eligible**. These are prototype rules only, not insurer policy or professional advice.

## Repository Contents

The repository view includes the following folders and files. Some names are shortened in the GitHub file browser, and the repository contains multiple versions and supporting archives.

- `AI_VEHICLE_DATASET/` — vehicle-dataset resources.
- `outputs/` — generated outputs and evaluation artefacts.
- `LOGGING_VERSIONS/` — versioned project materials and logs.
- `NEW_VEHICLE_DAMAGE_VERSIONS/` — alternative project versions.
- `NOVELTY_VERSIONS/` — additional versioned materials.
- `CMD RUNNING COMMMNDS-DL/` — command/setup notes.
- Additional project/version folders, archives, and model files visible in the repository.

Because the repository contains multiple versions, use the instructions and model checkpoints that belong to the specific version you intend to run. Do not assume that all checkpoints or datasets are interchangeable.

## Setup

### 1. Clone the repository with Git LFS

Large model and dataset files are stored using Git LFS. Install Git LFS before cloning or pulling the large files.

```bash
git lfs install
git clone <REPOSITORY_URL>
cd AI-Vehicle-Damage-Assessment-Final
git lfs pull
```

Replace `<REPOSITORY_URL>` with the repository's clone URL from GitHub.

If you already cloned the repository before installing Git LFS, run:

```bash
git lfs install
git lfs pull
```

### 2. Set up the Python backend

Use the Python version and dependencies specified by the selected backend version, if a requirements file is provided.

```bash
python -m venv .venv
```

Activate the environment:

**Windows (Command Prompt):**
```bat
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

If the selected backend directory contains `requirements.txt`, install its dependencies from that directory:

```bash
pip install -r requirements.txt
```

### 3. Set up the React frontend

Open a terminal in the selected frontend directory—the directory containing `package.json`—and run:

```bash
npm install
```

### 4. Start the application

The exact startup commands depend on the project version and the locations/names of its backend entry point and frontend scripts. Check the relevant version's README, command notes, `package.json`, and Python entry-point files for the correct commands. Start the backend and frontend in separate terminals if that version requires both services.

> **Important:** Python virtual environments, installed Python packages, and `node_modules` are not included in the repository. They must be recreated locally.

## Typical Application Outputs

Depending on the selected project version and the image passing the validation stages, the application can return:

- Image-authenticity status and confidence
- Photo-quality feedback
- Prediction-reliability feedback
- Binary damage-detection result
- Predicted damage class and confidence
- Segmentation mask or damage-region visualization
- Estimated damaged-area percentage
- Damage severity category
- Assessment score
- Project-defined insurance eligibility label

The exact output fields may differ between versions.

## Limitations

- Model performance depends on dataset coverage, image quality, lighting, viewpoint, vehicle type, and damage appearance.
- Results from the held-out test set do not guarantee performance on unrelated external images.
- Segmentation errors affect estimated damaged area and may affect severity categories.
- Image authenticity detection can produce false positives and false negatives; it is not forensic verification.
- Photo-quality and prediction-stability checks are advisory signals, not proof that a prediction is correct.
- Pixel-area percentage is not a direct estimate of repair cost or structural severity.
- Eligibility is generated by project-defined scoring rules and must not be presented as a real insurer's decision.

## Future Improvements

Potential next steps include evaluating on larger and more diverse independently labelled datasets, improving cross-dataset generalization, validating the authenticity detector against varied real and generated images, and independently checking the quality and reliability heuristics.

## Academic Project

**Title:** AI-Based Vehicle Damage Assessment and Insurance Eligibility Prediction Using Deep Learning  
**Institution:** Amrita Vishwa Vidyapeetham, Amrita School of Computing, Chennai  
**Project type:** Deep Learning end-semester academic project

---

*For reproducibility, use the model checkpoints, dependency files, and run instructions associated with the exact version being evaluated. Reported metrics in this README are taken from the project report and should be read with the limitations described above.*
