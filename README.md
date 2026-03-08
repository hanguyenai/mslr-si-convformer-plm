# Top-7 Solution for MSLR Task 1 (CSLR Track)

This repository contains our **Top-7 solution** for **MSLR Task 1 (Signer Independent)** in the Continuous Sign Language Recognition (CSLR) track.

Our approach is built upon a **ConvFormer-based architecture** for pose-based gloss recognition. Starting from a ConvFormer backbone that achieved **13.0652 WER**, we further improved the system by incorporating a **Pretrained Language Model (PLM)** and additional **data augmentation strategies**, achieving **9.41 WER** on the Signer Independent task.

---

## Overview

Continuous Sign Language Recognition requires modeling both fine-grained local motion patterns and long-range temporal dependencies from sign videos. Our framework is based on a **ConvFormer-style architecture**, which combines temporal convolutional modeling with Transformer-style sequence learning.

To further enhance performance, we extend the ConvFormer backbone with:

- a **Pretrained Language Model (PLM)** to provide stronger linguistic constraints for gloss sequence prediction,
- and additional **data augmentation techniques** to improve generalization and robustness.

With these improvements, our final system reduces the WER from **13.0652** to **9.41**.

---

## Result

| Task | Method | WER |
|------|--------|-----|
| Signer Independent | ConvFormer baseline | 13.0652 |
| Signer Independent | ConvFormer + PLM + Augmentation (ours) | **9.41** |

**Competition Ranking:** **Top 7** in MSLR Task 1.

---

## Key Contributions

- Built upon a **ConvFormer-based CSLR model** for pose-based sign language recognition.
- Improved the baseline with **PLM integration** for better sequence-level language modeling.
- Applied additional **augmentation strategies** to improve robustness and generalization.
- Achieved **9.41 WER**, improving over the ConvFormer baseline result of **13.0652**.

## Setup Instructions

Follow these steps to set up the environment and get started:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gufranSabri/Pose86K-CSLR-Isharah.git
   cd Pose86K-CSLR-Isharah
   ```

2. **Download the dataset** from [TASK 1](https://www.kaggle.com/datasets/gufransabri3/mslr-task1); [TASK 2](https://www.kaggle.com/datasets/gufransabri3/mslr-task2). Place the dataset in the `./data` folder.

3. **Set up the Python environment**:
   - Install `virtualenv`:
     ```bash
     pip install virtualenv
     ```

   - Create a virtual environment and activate it:
     ```bash
     python<version> -m venv pose
     source pose/bin/activate  # On Windows: pose\Scriptsctivate
     ```

   - Install the required dependencies:
     ```bash
     pip install torch==1.13 torchvision==0.14 tqdm numpy==1.23.5 pandas opencv-python
     git clone --recursive https://github.com/parlance/ctcdecode.git
     cd ctcdecode && pip install .
     ```


## Running the Model
Once your environment is ready and the data is in place, you can run the main script using the following format:
```
python main.py \
  --work_dir ./work_dir/test \
  --data_dir ./data \
  --mode SI \
  --model SOTA_CSLR_PLM \
  --device 0 \
  --lr 0.0001 \
  --num_epochs 300
```

### Argument Descriptions
 * ```--work_dir:``` Path to store logs and model checkpoints (default: ./work_dir/test)
 * ```--data_dir:``` Path to the dataset directory (default:``` /data/sharedData/Smartphone/)
 * ```--mode:``` Task mode, either SI (Signer Independent) or US (Unseen Sentences)
 * ```--model:``` Model variant to use (base, or any other available variant)
 * ```--device:``` GPU device index (default: 0)
 * ```--lr:``` Learning rate (default: 0.0001)
 * ```--num_epochs:``` Number of training epochs (default: 300)

You can modify these arguments as needed for your experiments.

### Example Command
```
python main.py --work_dir ./work_dir/base_US --model base --mode US
```

## Usage

Once the environment is set up, you can train or test the model on the available tasks. Follow the instructions in the individual task directories for specific commands.

## License

This project is licensed under the MIT License.
# mslr-si-convformer-plm
