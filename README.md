# Basic Streamlit Image Captioning App

## Acknowledgment
- I have implemented and modified the core tech - Captioning Module - based on AoA model 
- For extract feature tool (FasterRCNN): https://github.com/facebookresearch/mmf/tree/main
- For model architecture: (https://github.com/husthuaan/AoANet)

## Installation
```
!git clone https://github.com/21522173huy/Streamlit-App.git
!cd Streamlit-App
```

## Dependencies
- torch : 2.3.0
- streamlit : 1.35.0
- omegaconf : 2.3.0
- iopath : 0.1.10
- pytorch_lightning : 2.2.5
- lmdb : 1.4.1
- wget : 3.2
- pyngrok : 7.1.6
- gdown : 4.7.3

## Usage
- Note that i ran the repo in Google Collab, so just in case there would be any path erros please check: model_captioning.py, ultils.py and main.py
```
!wget -q -O - ipv4.icanhazip.com
! streamlit run main.py & npx localtunnel --port 8501
```

## DEMO
- Navigate to the [demo.ipynb](example/demo.ipynb) notebook.
- Follow the code to have a general background about how to use the project.

## Limitations and Future Work
### Limitations
- Dataset : The core tech trained on UIT-VIlC dataset, which is a relatively small and domain-specific dataset focused on sports images, particularly tennis. This means the model does not perform well on more diverse or general-purpose images
- Model architecture : The app implemented Attention on Attention (AoA) model. While AoA has performed well, there are many benchmark powerful image-captioning models.

### Future Work
- Trying to apply the model on a new dataset that is better in terms of quality and quantity
- Researching and Implementating more powerful architectures, Transformer-based for instance, to achieve better captioning performance 
- Enhacing Steamlit app interface to be more nice-looking and professional
