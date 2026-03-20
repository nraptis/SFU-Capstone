# SFU Capstone

![SFU Capstone banner](https://raw.githubusercontent.com/nraptis/SFU-Capstone/refs/heads/main/11_deployment_implementation/ui/images/banner.jpg)

This repository contains our SFU capstone work on blood-cell microscopy classification, model experimentation, and deployment.

The majority of the project lives in `07_and_08_experiment_and_scale` and `11_deployment_implementation`. Of those two, `07_and_08_experiment_and_scale` represents about 65% of the total effort and contains most of the experimentation, tuning, and scaling work that shaped the final system.

The final `iguana64` + `falcon64` ResNet suite performs very well within the domain it was trained on, but generalizes poorly across microscopy images outside that domain.

The live deployment is available at [https://secretbloodtest.com/](https://secretbloodtest.com/) and should be referenced without `www`.

<p align="center">
  <img src="https://raw.githubusercontent.com/nraptis/SFU-Capstone/refs/heads/main/you_might_be_super_sick.jpg" alt="Blood cell image" width="520">
</p>

## Project Map

- `07_and_08_experiment_and_scale`: the core of the project, including model fights, preprocessing tests, augmentation work, hyperparameter tuning, and final model selection.
- `11_deployment_implementation`: the production-facing implementation, including the inference pipeline, Flask app, UI assets, and deployment materials.
- The remaining folders document proposal work, data collection, benchmarking, deployment prep, and the final live handoff.

## Run The App

The local Flask app lives in `11_deployment_implementation`.

### 1. Install Anaconda

Download and install Anaconda from [https://www.anaconda.com/download](https://www.anaconda.com/download).

### 2. Make A Conda Environment

```bash
conda create -n sfu-capstone python=3.10 -y
conda activate sfu-capstone
```

### 3. Install VS Code

Download and install VS Code from [https://code.visualstudio.com/](https://code.visualstudio.com/).

### 4. Navigate To The Project Folder

```bash
cd /path/to/SFU-Capstone/11_deployment_implementation
```

Optional: open the folder in VS Code.

```bash
code .
```

### 5. Install `requirements.txt`

```bash
pip install -r requirements.txt
```

### 6. Exact Flask Commands

Install requirements:

```bash
pip install -r requirements.txt
```

Install Flask explicitly:

```bash
pip install flask
```

Configure Flask:

```bash
export FLASK_APP=app.py
export FLASK_DEBUG=1
```

Run Flask:

```bash
flask run --host=0.0.0.0 --port=8080
```

Navigate to the local site:

```text
http://127.0.0.1:8080/
```
