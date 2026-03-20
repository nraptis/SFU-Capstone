# SFU Capstone

![SFU Capstone banner](https://raw.githubusercontent.com/nraptis/SFU-Capstone/refs/heads/main/11_deployment_implementation/ui/images/banner.jpg)

This repository contains our SFU capstone work on blood-cell microscopy classification, model experimentation, and deployment.

The majority of the project lives in `07_and_08_experiment_and_scale` and `11_deployment_implementation`. Of those two, `07_and_08_experiment_and_scale` represents about 65% of the total effort and contains most of the experimentation, tuning, and scaling work that shaped the final system.

The final `iguana64` + `falcon64` ResNet suite performs very well within the domain it was trained on, but generalizes poorly across microscopy images outside that domain.

<p align="center">
  <img src="https://raw.githubusercontent.com/nraptis/SFU-Capstone/refs/heads/main/you_might_be_super_sick.jpg" alt="Blood cell image" width="520">
</p>

## Project Map

- `07_and_08_experiment_and_scale`: the core of the project, including model fights, preprocessing tests, augmentation work, hyperparameter tuning, and final model selection.
- `11_deployment_implementation`: the production-facing implementation, including the inference pipeline, Flask app, UI assets, and deployment materials.
- The remaining folders document proposal work, data collection, benchmarking, deployment prep, and the final live handoff.
