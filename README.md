# RMS-UNet: Residual Multi-Scale UNet for Liver and Lesion Segmentation

This repository provides the research implementation associated with:

**Khan RA, Luo Y, Wu FX. RMS-UNet: Residual multi-scale UNet for liver and lesion segmentation. _Artificial Intelligence in Medicine_. 2022;124:102231.**  
https://doi.org/10.1016/j.artmed.2021.102231

RMS-UNet was proposed for automated segmentation of the liver and liver lesions from medical images. The repository contains the proposed neural network architecture, the data preprocessing and train/validation/test pipeline, and training scripts corresponding to different loss-function variants investigated during the study.

---

## Repository Contents

The major files in this repository are organized as follows:

| File / Pattern | Description |
|---|---|
| `ResU*` | Implementation of the proposed **RMS-UNet / residual multi-scale UNet architecture** used in the study. |
| `Data_TVT*` | Data preprocessing and preparation pipeline, including organization of the data into **training, validation, and testing** components. |
| `main*` | Main training and experimentation scripts. Different `main*` files correspond to different **loss-function variants and experimental configurations** investigated during model development. |

The exact experimental configuration should be selected according to the loss function or experiment being reproduced.

---

## RMS-UNet

RMS-UNet was developed to improve automated liver and liver-lesion segmentation by incorporating residual learning and multi-scale feature representation within a U-Net-based segmentation framework.

The implementation provided here represents the research code used during the development and evaluation of the methodology described in the associated publication.

For full information regarding the network architecture, experimental design, datasets, evaluation methodology, loss functions, and reported results, users should refer to the original article:

> Khan RA, Luo Y, Wu FX. RMS-UNet: Residual multi-scale UNet for liver and lesion segmentation. *Artificial Intelligence in Medicine*. 2022;124:102231.  
> DOI: https://doi.org/10.1016/j.artmed.2021.102231

---

## General Workflow

The scripts follow the general experimental workflow:

```text
Input Medical Images
        |
        v
Data Preprocessing
(Data_TVT)
        |
        v
Train / Validation / Test Preparation
        |
        v
RMS-UNet Architecture
(ResU)
        |
        v
Training using selected loss-function variant
(main*)
        |
        v
Model Evaluation
        |
        v
Liver / Lesion Segmentation
```

Different `main*` scripts may implement different loss functions or experimental settings while using the same underlying RMS-UNet architecture and preprocessing framework.

---

## Usage

The repository is intended primarily for:

- reproducibility of the published RMS-UNet research;
- academic research;
- comparison with other medical-image segmentation approaches;
- methodological development;
- educational use; and
- extension of the proposed RMS-UNet architecture.

Users should review the individual scripts before execution and modify dataset paths, directory structures, training parameters, and other environment-specific settings as required.

The repository contains research code and should not be assumed to operate as a turn-key clinical software package.

---

## Data

Medical imaging datasets are **not redistributed through this repository unless explicitly stated otherwise**.

Users are responsible for obtaining appropriate datasets independently and complying with the corresponding dataset licenses, data-use agreements, institutional requirements, privacy requirements, and ethical approvals.

Please refer to the published RMS-UNet article for details regarding the datasets and experimental methodology used in the original study.

---

## Citation

### Required citation for use of the RMS-UNet implementation

If you use, modify, adapt, extend, benchmark, or incorporate this code or the RMS-UNet architecture in **research that results in a publication, preprint, thesis, dissertation, conference paper, report, or other scholarly output**, cite the following article:

```bibtex
@article{khan2022rmsunet,
  title={RMS-UNet: Residual multi-scale UNet for liver and lesion segmentation},
  author={Khan, Rayyan A and Luo, Yong and Wu, Fang-Xiang},
  journal={Artificial Intelligence in Medicine},
  volume={124},
  pages={102231},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.artmed.2021.102231}
}
```

Full citation:

**Khan RA, Luo Y, Wu FX. RMS-UNet: Residual multi-scale UNet for liver and lesion segmentation. Artificial Intelligence in Medicine. 2022;124:102231.**

https://doi.org/10.1016/j.artmed.2021.102231

Citation of the corresponding publication helps support reproducible research and allows appropriate attribution to the original methodology.

---

## Related Publication

The following systematic review provides broader background on machine-learning approaches to liver-disease diagnosis:

**Khan RA, Luo Y, Wu FX. Machine learning based liver disease diagnosis: A systematic review. _Neurocomputing_. 2022;468:492–509.**

https://doi.org/10.1016/j.neucom.2021.08.138

```bibtex
@article{khan2022liverreview,
  title={Machine learning based liver disease diagnosis: A systematic review},
  author={Khan, Rayyan A and Luo, Yong and Wu, Fang-Xiang},
  journal={Neurocomputing},
  volume={468},
  pages={492--509},
  year={2022},
  publisher={Elsevier},
  doi={10.1016/j.neucom.2021.08.138}
}
```

The systematic review is provided as related background literature and should not be interpreted as the primary publication describing the code contained in this repository.

---

## Terms of Use

This repository and its source code are provided primarily for **academic, educational, and non-commercial research purposes**.

Use of the code is subject to the terms specified in the repository's `LICENSE` file.

### Non-commercial research use

The code may be used for permitted non-commercial research and educational activities subject to the applicable license conditions.

Researchers using this implementation in scholarly work are required to provide appropriate attribution and cite the RMS-UNet publication identified above.

### Commercial use

**Commercial use is not authorized without prior written permission from the authors/rightsholders.**

This includes, where applicable, use of the software or substantial portions of it in:

- commercial products;
- commercial software or services;
- paid clinical systems;
- commercial decision-support systems;
- proprietary products developed for commercial exploitation; or
- other activities primarily intended for commercial advantage.

Organizations or individuals interested in commercial use should contact the authors/rightsholders to discuss appropriate authorization or licensing.

No commercial rights should be inferred merely from public availability of the source code.

---

## Modification and Redistribution

Any permitted modification or redistribution of this software must comply with the accompanying license.

Where redistribution is permitted, attribution to the original work should be preserved, including:

- identification of the original authors;
- reference to this repository;
- reference to the RMS-UNet publication; and
- preservation of applicable copyright and license notices.

Modified versions should be clearly identified as modifications and should not be represented as the original implementation produced or validated by the authors.

---

## Research and Clinical Disclaimer

This repository contains **research software** developed for scientific investigation.

It is **not a medical device** and has not been provided as a validated clinical diagnostic, treatment-planning, or medical decision-making system.

The software should not be used as a substitute for professional medical judgment or as the sole basis for clinical decisions.

Users are responsible for independently validating the software and ensuring compliance with all applicable regulatory, ethical, privacy, institutional, and legal requirements before using it in any medical or clinical environment.

---

## No Warranty

The software is provided on an **"AS IS"** basis, without warranties or guarantees regarding accuracy, performance, reliability, fitness for a particular purpose, clinical suitability, or reproducibility in environments different from those used in the original research.

The authors shall not be responsible for losses, damages, clinical outcomes, or other consequences arising from the use or misuse of the software to the extent permitted by applicable law.

---

## Third-Party Software and Data

This repository may depend on external software libraries, frameworks, datasets, or other third-party resources.

Such third-party materials remain subject to their respective copyrights, licenses, terms of use, and data-use requirements.

The license governing this repository does not grant rights to any third-party software, datasets, medical images, or other materials that may be required to reproduce the experiments.

---

## Reproducibility

The code is released to improve transparency and reproducibility of the research described in the RMS-UNet publication.

Because software libraries, computing environments, hardware, datasets, preprocessing procedures, and dependency versions may change over time, exact numerical reproduction of the originally reported results cannot be guaranteed without recreating the corresponding experimental environment.

Users are encouraged to consult the publication for the experimental methodology and report any substantial modifications when publishing results obtained using this repository.

---

## Reporting Issues

Researchers who identify implementation issues, ambiguities, or potential improvements are encouraged to submit a GitHub issue.

When reporting an issue, include sufficient information to reproduce the problem, including relevant script names, configuration details, software environment, and error messages where appropriate.

---

## Acknowledgment

If this repository contributes to your research, we appreciate appropriate acknowledgment of the original work through citation of:

**Khan RA, Luo Y, Wu FX. RMS-UNet: Residual multi-scale UNet for liver and lesion segmentation. Artificial Intelligence in Medicine. 2022;124:102231.**

DOI: https://doi.org/10.1016/j.artmed.2021.102231

---

## Authors

**Rayyan A. Khan**  
**Yong Luo**  
**Fang-Xiang Wu**

For questions regarding the methodology, research use, or potential commercial licensing, contact the corresponding author(s) through the contact information provided in the original publication or through the repository contact information.

---

## License

Copyright © RMS-UNet authors.

This repository is made available subject to the terms contained in the accompanying `LICENSE` file.

**Academic/non-commercial use and commercial use are treated differently. Commercial use requires appropriate prior authorization where specified by the license.**

Users should read the complete `LICENSE` file before copying, modifying, distributing, or otherwise using the source code.
