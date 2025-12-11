# Dense Policy: Bidirectional Autoregressive Learning of Actions

[![Paper on arXiv](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2503.13217) [![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://selen-suyue.github.io/DspNet/) [![Tasks Report](https://img.shields.io/badge/Report-PDF-orange.svg)]()

**Authors**: <a href="https://selen-suyue.github.io" style="color: maroon; text-decoration: none; font-style: italic;">Yue Su*</a><sup></sup>,
<a href="https://scholar.google.com/citations?user=WurpqEMAAAAJ&hl=en" style="color: maroon; text-decoration: none; font-style: italic;">Xinyu Zhan*</a><sup></sup>,
<a href="https://tonyfang.net/" style="color: maroon; text-decoration: none; font-style: italic;">Hongjie Fang</a>,
<a href="https://hanxue.me/" style="color: maroon; text-decoration: none; font-style: italic;">Han Xue</a>,
<a href="https://fang-haoshu.github.io/" style="color: maroon; text-decoration: none; font-style: italic;">Haoshu Fang</a>,
<a href="https://dirtyharrylyl.github.io/" style="color: maroon; text-decoration: none; font-style: italic;">Yong-Lu Li</a>,
<a href="https://www.mvig.org/" style="color: maroon; text-decoration: none; font-style: italic;">Cewu Lu</a>,
<a href="https://lixiny.github.io/" style="color: maroon; text-decoration: none; font-style: italic;">Lixin Yang&dagger;</a><sup></sup>

![teaser](assets/images/teaser.png)
## 🛫 Getting Started
This is the 3D version of Dense Policy, you can also refer [2D Dense Policy Code](https://github.com/Selen-Suyue/DensePolicy2D) here.

You can also refer [DSPv2](https://github.com/Selen-Suyue/DSPv2), an effective and generalizable dense policy, can deploy for whole-body mobile manipulation.
### 💻 Installation

Please following the [installation guide](assets/docs/INSTALL.md) to install the `dsp` conda environments and the dependencies, as well as the real robot environments. Also, remember to adjust the constant parameters in `dataset/constants.py` and `utils/constants.py` according to your own environment.

### 📷 Calibration

Please calibrate the camera(s) with the robot before data collection and evaluation to ensure correct spatial transformations between camera(s) and the robot. Please refer to [calibration guide](assets/docs/CALIB.md) for more details.

### 🛢️ Data Collection

You can view the sampled data (cut task) from this [link](https://drive.google.com/drive/folders/15vCT9F_s6qdLfm-Hv22WhZTHv1KsjDa1?usp=drive_link), which contains task data (one trajectory for instance). You can ignore other files
since they are for [MBA](https://selen-suyue.github.io/MBApage/)
We follow the data managemnet pattern as [RH20T](https://rh20t.github.io/).

```
Task_name
`-- train/
    |-- [episode identifier 1]
    |   |-- metadata.json              # metadata
    |   |-- timestamp.txt              # calib timestamp  
    |   |-- cam_[serial_number 1]/    
    |   |   |-- color                  # RGB
    |   |   |   |-- [timestamp 1].png
    |   |   |   |-- [timestamp 2].png
    |   |   |   |-- ...
    |   |   |   `-- [timestamp T].png
    |   |   |-- depth                  # depth
    |   |   |   |-- [timestamp 1].png
    |   |   |   |-- [timestamp 2].png
    |   |   |   |-- ...
    |   |   |   `-- [timestamp T].png
    |   |   |-- tcp                    # tcp
    |   |   |   |-- [timestamp 1].npy
    |   |   |   |-- [timestamp 2].npy
    |   |   |   |-- ...
    |   |   |   `-- [timestamp T].npy
    |   |   `-- gripper_command        # gripper command
    |   |       |-- [timestamp 1].npy
    |   |       |-- [timestamp 2].npy
    |   |       |-- ...
    |   |       `-- [timestamp T].npy
    |   `-- cam_[serial_number 2]/     # similar camera structure
    `-- [episode identifier 2]         # similar episode structure
```


### 🧑🏻‍💻 Training

```bash
conda activate dsp
bash train.sh
```

### 🤖 Evaluation

Please follow the [deployment guide](assets/docs/DEPLOY.md) to modify the evaluation script.

Modify the arguments in `eval.sh`, then

```bash
conda activate dsp
bash eval.sh
```

## ✍️ Citation

```bibtex
@InProceedings{Su_2025_ICCV,
    author    = {Su, Yue and Zhan, Xinyu and Fang, Hongjie and Xue, Han and Fang, Hao-Shu and Li, Yong-Lu and Lu, Cewu and Yang, Lixin},
    title     = {Dense Policy: Bidirectional Autoregressive Learning of Actions},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {14486-14495}
}
```

## 📃 License

<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="" rel="cc:attributionURL" href="https://selen-suyue.github.io/DspNet/">DSP</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1" alt=""></a></p>
