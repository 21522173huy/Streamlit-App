
import streamlit as st
from ultils import set_background, load_components, process_sequence, extract_feature, predict_caption
from PIL import Image
import numpy as np
import torch
import os

@st.cache_resource
def load_model_and_components():
    with st.spinner('Loading model and components...'):
        model_caption, tokenizer, frcnn, frcnn_config, image_preprocess = load_components()
    return model_caption, tokenizer, frcnn, frcnn_config, image_preprocess

model_caption, tokenizer, frcnn, frcnn_config, image_preprocess = load_model_and_components()

uploads_dir = 'upload_images'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

set_background('background/bg5.png')

# set title
st.title('Sport Images Captioning')

# set header
st.header('Please upload an image')

image_placeholder = st.empty()
caption_placeholder = st.empty()

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# display image
if file is not None:

    file_url = f'{uploads_dir}/{file.name}'
    file_path = os.path.join(uploads_dir, file.name)

    image = Image.open(file).convert('RGB')
    image.save(file_path)

    with image_placeholder.container():
        st.image(image, use_column_width=True)

    # generate caption
    caption = predict_caption(model_caption, frcnn, frcnn_config, image_preprocess, f'/content/Streamlit-App/{file_url}', tokenizer, device)

    # write caption
    with caption_placeholder.container():
        st.write("## {}".format(caption))
