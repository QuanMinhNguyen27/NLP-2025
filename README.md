Introduction

This ASR (Automatic Speech Recognition) project utilizes a CNN + BiLSTM + CTC Loss model to convert speech into text. The system is built using TensorFlow/Keras, supports GPU training, and allows exporting the model to ONNX for faster inference.

 1️⃣ Environment Setup

The project requires Python 3.8+ and the following libraries:

TensorFlow

Keras

NumPy

Pandas

Librosa

OpenCV

Scipy

ONNX

Tqdm

Matplotlib

Jiwer

PyTorch (optional for certain conversions)

Google Drive (if using Colab for training)

If running on Google Colab, you may also need to install additional dependencies such as unrar for dataset extraction.

 2️⃣ Download Training Data

The dataset used is LJSpeech-1.1. You can download it either from Google Drive (if pre-uploaded) or directly from an online source.

Ensure that the dataset is extracted and placed in the correct directory before training.

 3️⃣ Train the Model

Run the training script to start model training. The system will:

Load data from dataProvider.py

Preprocess data using preprocessing.py

Build the model using model.py

Use CTC Loss (losses.py) for sequence output processing

Compute CER & WER for model evaluation (metrics.py)

Utilize callbacks (callbacks.py) for logging and model saving

Save model checkpoints to checkpoints/

Training time depends on dataset size and the number of epochs.

📌 4️⃣ Test the Trained Model

After training, run the inference script to check results. The model should take an audio file as input and return the corresponding text.

Additionally, test scripts are provided to verify the correctness of the model's output.

 5️⃣ Convert Model to ONNX (Faster Inference)
Once training is complete, convert the model to ONNX format for optimized inference. Save the trained model for future use or deployment.

 6️⃣ Project Directory Structure

ASR_Project/
│── Datasets/
│   │── LJSpeech-1.1/
│   │   │── wavs/                    # Audio files
│   │   │── metadata.csv              # Transcription metadata
│   │   └── README                    # Dataset documentation
│
│── sound_to_text/
│   │── configs.py                     # Model configurations
│   │── inferenceModel.py               # Inference script
│   │── model.py                        # ASR Model (CNN + BiLSTM + CTC Loss)
│   │── train.py                        # Training script
│
│── tensorflow/
│   │── callbacks.py                     # Training callbacks
│   │── dataProvider.py                  # Data loading & processing
│   │── layers.py                        # Custom model layers
│   │── losses.py                        # CTC Loss function
│   │── metrics.py                       # Evaluation metrics (CER, WER)
│   │── model_utils.py                   # Model utility functions
│
│── Tests/
│   │── test_tensorflow_metrics.py       # Test TensorFlow metric functions
│   │── test_text_utils.py               # Test text processing functions
│
│── checkpoints/                         # Stores trained model checkpoints
│── README.md                            # Project documentation

7️⃣ References

[TensorFlow ASR Guide](https://www.tensorflow.org/resources/models-datasets)
https://pypi.org/project/mltu/
[LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)# nlp
