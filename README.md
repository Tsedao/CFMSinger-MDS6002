# CFMSinger: Improve Singing Voice Conversion via Fast Conditional Flow Matching
[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) [![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-blue.svg)](https://pytorch.org/) [![Hugging Face Transformers](https://img.shields.io/badge/Transformers-4.11.3-brightgreen.svg)](https://huggingface.co/transformers/) ![OpenAI Whisper Badge](https://img.shields.io/badge/OpenAI-Whisper-8f00b3.svg) [![Librosa](https://img.shields.io/badge/librosa-orange.svg)](https://librosa.org/)


CFMSinger is a deep learning model that uses a conditional flow matching approach to improve the quality and sample speed of a many-to-many singing voice conversion. The model is intended to outperform two baseline models based on GAN and Diffusion vocoder in terms of audio quality and inference speed from a source singer to the target singer.


## Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/CFMSinger.git
cd CFMSinger
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Model Architecture
The proposed model architecture is shown in the following diagram:
![Model Architecture](assets/nlp.png)

## Usage
### Config and Preprocess
```bash
cd preprocess
python processs_opencpop.py # extract train json
python process_m4singer.py  # extract test json
python extract_whisper.py   # extract ppg
python extract_lf0.py
python extract_hubert.py    # extract phoneme
```


### Training
To train the GAN and Diffusion vocoder baseline models, run:
```bash
python train_gansvc.py
python train_diffsvc.py
```

To train the CFMSinger model run:
```bash
python train_cfmsvc.py
```

### Inference
To generate audio using a trained model, run:
```bash
python inference.py --model_path path/to/model --input_path path/to/input --output_path path/to/output
```

## Results
The following figures show the results of our experiments:

### Audio Samples
![Audio Samples](assets/tb_audio.png)
<!-- | Audio File | Groundtruth |  Generated  | Conversion  |
|------------|-------------|-------------|-------------|
| HIFI-GAN   | <audio controls>
  <source src="assets/gt_clip3.wav" type="audio/wav">
</audio> |<audio controls>
  <source src="assets/GAN_pred_clip3.wav" type="audio/wav">
</audio>| |
| DiffWave   |            | | |
| CFM        |            | | | -->

### Mel Spectrograms
![Mel Spectrograms](assets/tb_mel.png)

### Scalar Values
![Scalar Values](assets/tb_scalar.png)

## License
This project is licensed under the MIT [License](LICENSE) - see the LICENSE file for details.
