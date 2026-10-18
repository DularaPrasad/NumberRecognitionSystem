# Handwritten Number Recognition System (HNRS)

COS30018 Intelligent Systems, Swinburne University of Technology. Group project (Option B).

A system that reads handwritten numbers from an image. The pipeline has four stages:

1. **Preprocessing** (Task 1): clean up the input image (grayscale, resize, binarise).
2. **Segmentation** (Task 2): split an image of a number into one image per digit.
3. **Digit recognition** (Task 3): classify each digit (0 to 9) with a trained model.
4. **Integration and evaluation** (Task 4), plus a **GUI**: combine the stages and test the whole system.

## Repository structure

```
NumberRecognitionSystem/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── preprocessing/    Task 1: image preprocessing
│   ├── segmentation/     Task 2: splitting a number into digits
│   ├── models/           Task 3: digit recognition models
│   │   ├── baseline_knn.py
│   │   ├── cnn_model.py
│   │   ├── mlp_model.py
│   │   ├── lenet5_model.py
│   │   └── resnet_model.py
│   └── exploration/      load_mnist.py: inspect the MNIST data
│   └── evaluation/       plot_training_curves.py
├── results/              plots saved by the scripts (confusion matrices, training curves)
└── saved_models/         trained model weights (.pth)
```

`data/` (the downloaded MNIST files) and `venv/` (the virtual environment) are created locally and are not committed.

## Team and status

| Component | Folder | Owner | Status |
|---|---|---|---|
| Task 1: Preprocessing | `src/preprocessing` | Declan | Complete |
| Task 2: Segmentation | `src/segmentation` | Charith | Complete |
| Task 3: Digit recognition models | `src/models` | Dulara | Complete |
| Task 4: Integration and evaluation | | Dulara | Pending |
| GUI | | Declan | Pending |

## Setup (Windows)

Python 3.12 is recommended, because every library used here supports it (PyTorch, TensorFlow, OpenCV, scikit-learn).

```
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The models also train on a CPU. To use an NVIDIA GPU, install the CUDA build of PyTorch from pytorch.org instead of the default one.

## Task 3: digit recognition models

Four models are trained and compared on MNIST (60,000 training images, 10,000 test images). Run any of them with the virtual environment active:

```
python src\models\cnn_model.py
python src\models\mlp_model.py
python src\models\lenet5_model.py
python src\models\resnet_model.py
```

The first run downloads MNIST into `data/`. Each script prints its parameter count, per-epoch loss, test accuracy and per-digit accuracy, then saves a confusion matrix and a training curve to `results/` and the trained weights to `saved_models/`. The scripts work from any folder.

### Results

| Model | Parameters | Test accuracy | Training time |
|---|---|---|---|
| CNN | 20,490 | 98.75% | 46.7 s |
| MLP | 109,386 | 97.84% | 32.6 s |
| LeNet-5 | 61,706 | 98.87% | 35.2 s |
| ResNet | 77,754 | **99.22%** | 53.3 s |

All models: 10 epochs, batch size 64, Adam (learning rate 0.001), full training set, one run each. Training times were measured on an NVIDIA RTX 4080.
The CNN's saved model comes from a repeat of the 10-epoch run (98.75%); its first run scored 98.96%. The training time is from the first run.

- **Convolution is the dividing line.** The MLP trails the three convolutional models by about 0.9 to 1.4 points, even though it has the most parameters.
- **The ResNet is the most accurate.** It is 0.47 points above the saved CNN (0.26 above the CNN's first run). Identical CNN runs differ by about 0.2 points, so the ResNet is above both, but each model was run once. Treat the ranking as provisional until the models are tested on real handwritten photos.
- **The KNN baseline (`baseline_knn.py`) is not part of the comparison.** It was used as an early check that the pipeline worked, and was replaced by the models above on the tutor's advice.

### Saved models

`saved_models/` holds each model's trained weights (PyTorch `state_dict` files). The model classes are currently defined inside each script, and each script trains when it runs. For integration (Task 4) they will be split into importable modules, so the trained models can be loaded without retraining.

## References

- LeCun, Y., Bottou, L., Bengio, Y. and Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324.
- He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*. arXiv:1512.03385.
- LeCun, Y., Cortes, C. and Burges, C. The MNIST database of handwritten digits.
