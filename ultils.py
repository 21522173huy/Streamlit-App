
import os
from model_caption import AoA_Model

from tools.scripts.features.frcnn.frcnn_utils import Config
from tools.scripts.features.frcnn.modeling_frcnn import GeneralizedRCNN
from tools.scripts.features.frcnn.processing_image import Preprocess
from collections import Counter
from transformers import AutoTokenizer
import gdown
import wget
import base64
import streamlit as st
import torch

def set_background(image_path):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_path, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)

def load_components(device = 'cpu',
                   extract_config_file = "https://s3.amazonaws.com/models.huggingface.co/bert/unc-nlp/frcnn-vg-finetuned/config.yaml",
                   extract_config_weight = 'https://s3.amazonaws.com/models.huggingface.co/bert/unc-nlp/frcnn-vg-finetuned/pytorch_model.bin',
                   predict_caption_weight = "https://drive.google.com/uc?id=1nk1a-r1hsem2asIKVMSDSfOaVQYF5jIP",
                   language_name = "vinai/phobert-base-v2",
                  ):

  download_dir = 'downloaded_weights'
  if not os.path.exists(download_dir):
      os.makedirs(download_dir)

  config_path = os.path.join(download_dir, 'config.yaml')
  extract_path = os.path.join(download_dir, 'pytorch_model.bin')
  caption_weight_path = os.path.join(download_dir, 'caption_weight.pth')

  # Load Extract Features Model
  # Download the model config and weight file using wget

  if not os.path.exists(config_path) or not os.path.exists(extract_path) or not os.path.exists(caption_weight_path):
    wget.download(extract_config_file, config_path)
    wget.download(extract_config_weight, extract_path)
    gdown.download(predict_caption_weight, caption_weight_path)

  # Load the downloaded model config and weight file using from_pretrained
  frcnn_cfg = Config.from_pretrained(config_path)
  frcnn = GeneralizedRCNN.from_pretrained(extract_path, config=frcnn_cfg)
  image_preprocess = Preprocess(frcnn_cfg)

  # Load Predict Caption Model
  tokenizer = AutoTokenizer.from_pretrained(language_name)
  model_caption = AoA_Model(language_name = language_name,
                  features_size = 2048,
                  device = device,
                  )

  model_caption.load_state_dict(torch.load(caption_weight_path, map_location = torch.device(device)))

  return model_caption, tokenizer, frcnn, frcnn_cfg, image_preprocess

def process_sequence(sequence):
    tokens = sequence.split()
    counts = Counter(tokens)

    result = []
    for token in tokens:
        if counts[token] < 2 or token not in result:
            result.append(token)

    return ' '.join(result)

def extract_feature(frcnn, frcnn_cfg, image_preprocess, image_path, device):

  images, sizes, scales_yx = image_preprocess(image_path)

  output_dict = frcnn(
      images.to(device),
      sizes,
      scales_yx=scales_yx,
      padding=None,
      max_detections=frcnn_cfg.max_detections,
      return_tensors="pt",
  )

  return output_dict

def predict_caption(model, frcnn, frcnn_cfg, image_preprocess, image_path, tokenizer, device, max_length = 50):

  model.to(device), model.eval()
  frcnn.to(device), frcnn.eval()

  output_dict = extract_feature(frcnn, frcnn_cfg, image_preprocess, image_path, device)

  obj_features = output_dict['roi_features'].to(device)

  # Refining Image
  image_features = model.refiner_layer(obj_features).mean(dim = 1) # 36, 2048

  # Generate Answer
  count = 0

  captions_ids = torch.tensor([tokenizer.bos_token_id]).to(device) # Decoder Start Token

  # Constant Variables
  batch_size = 1
  embedding_size = image_features.size(-1)

  while count <= max_length:

    sequence_length = captions_ids.size(0)

    features_ = image_features.unsqueeze(dim = 1).expand(batch_size, sequence_length, embedding_size)

    embedded_captions = model.decoder_layer.embedding_layer(captions_ids.unsqueeze(dim = 0))

    input_concat = torch.cat([features_, embedded_captions], dim = 2)
    output, _ = model.decoder_layer.att_lstm(input_concat)

    att = model.decoder_layer.multi_head(output, image_features, image_features)

    residual_aoa = model.decoder_layer.residual_connect(output, att)

    output = model.decoder_layer.out_dropout(model.decoder_layer.out_linear(residual_aoa))

    predicted_token = output.argmax(dim = -1)

    captions_ids = torch.cat([captions_ids, predicted_token[:, -1]], dim = -1)

    count += 1

  res = tokenizer.decode(captions_ids, skip_special_tokens = True)
  return process_sequence(res)
