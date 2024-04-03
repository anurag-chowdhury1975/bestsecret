import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
import math
import requests
from io import BytesIO
from PIL import Image
import gradcam as gcam # Helper file contains the class definition for GradCAM
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.data import Dataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from struct import unpack
# from tqdm import tqdm

st.set_page_config(page_title="Image View Classifier", layout="wide")

@st.cache
def load_model_from_gdrive(url):
    response = requests.get(url)
    model_file = BytesIO(response.content)
    model = load_model(model_file)
    return model
    
def load_models():
    models = {}
    # model_bag = load_model('models/bag_resnet50_model_ft_all_93%.h5', custom_objects={'imagenet_utils': imagenet_utils})
    # model_clothes = load_model('models/clothes_resnet50_func_model_97%.h5', custom_objects={'imagenet_utils': imagenet_utils})
    # model_schuhe = load_model('models/schuhe_resnet50_model_ft_all_94%.h5', custom_objects={'imagenet_utils': imagenet_utils})
    # model_waesche = load_model('models/waesch_funcResnet_model_94%.h5', custom_objects={'imagenet_utils': imagenet_utils})
    model_bag = load_model_from_gdrive('https://drive.google.com/file/d/1VSitaSvcEuzNIPI_Mb1lIk9N4hBYVBWb/view?usp=sharing')
    model_clothes = load_model_from_gdrive('https://drive.google.com/file/d/1oCca1FE8YkAwo3GSglCCNWABdVnZdntX/view?usp=sharing')
    model_schuhe = load_model_from_gdrive('https://drive.google.com/file/d/1K83mAjX2mgp3uRaZ9-KjboXdjo6gq6k7/view?usp=sharing')
    model_waesche = load_model_from_gdrive('https://drive.google.com/file/d/1rMPdC4mGvUQ8JMreQnyP5GP_tcRqTsTq/view?usp=sharing')
    models['bag'] = model_bag
    models['clothes'] = model_clothes
    models['schuhe'] = model_schuhe
    models['waesche'] = model_waesche
    return models

def image_processing_function(im_path, input_img_dims, pre_process_function=None):

    orig = image.load_img(im_path)
    orig_arr = image.img_to_array(orig).astype("uint8")
    img = image.load_img(im_path, target_size=input_img_dims)

    image_arr = image.img_to_array(img)
    image_arr = np.expand_dims(image_arr, axis=0)

    return img, image_arr, orig_arr

def eval_model_on_test(model, test_ds):

    test_labels = []
    predictions = []

    # for imgs, labels in tqdm(test_ds.take(1000),
    #                          desc='Predicting on Test Data'):
    for imgs, labels in test_ds.take(1000):
        batch_preds = model.predict(imgs)
        predictions.extend(batch_preds)
        test_labels.extend(labels)
    if len(predictions[0]) > 1:
        predictions_max = np.argmax(predictions, axis=1)
    else:
        predictions_max = np.array(predictions)

    test_labels = np.array(test_labels)

    return test_labels, predictions_max, predictions

def get_mismatches(y_true, y_pred, BATCH_SIZE):
    num_mismatches = 0
    mismatch_tensor_indexes = {}
    for i in range(len(y_true)):
      if y_true[i] != y_pred[i]:
        num_mismatches += 1
        key = (i//BATCH_SIZE)
        tensor_index = (i % BATCH_SIZE)
        if mismatch_tensor_indexes.get(key) is not None:
          mismatch_tensor_indexes[key].append((tensor_index, i))
        else:
          mismatch_tensor_indexes[key] = [(tensor_index, i)]
    return num_mismatches, mismatch_tensor_indexes

models = load_models()
BATCH_SIZE = 32

model = ""

prod_labels = {
    "bag" : "Bags",
    "clothes" : "Clothes",
    "schuhe" : "Schuhe",
    "waesche" : "Waesche"
}

view_labels = {
    "bag": ["Front","Side","Inside","Back","Look"],
    "clothes": ["Model Front","Zoomed","Model Back","Ghost","Look"],
    "schuhe": ["Overll to Right","Back","Top or Sole","Side to Left","Zoom"],
    "waesche": ["Model Front","Zoomed","Model Back","Ghost","Look"]
}

def format_func(option):
    return prod_labels[option]


st.header("Image View Classifier")
prod_cat = st.selectbox(label="Select Product Category", options=list(prod_labels.keys()), format_func=format_func)
model = models.get(prod_cat)

prod_cat_class = st.selectbox(label="Select Product View", options=view_labels.get(prod_cat))

input_img_dims = (427, 350)

with st.form("my-form", clear_on_submit=True):
    uploaded_images = st.file_uploader("Please choose image(s)...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Predict Image View!")
    if submitted:
        if len(uploaded_images) > 0:
            # st.markdown('''
            #     <style>
            #         .uploadedFile {display: none}
            #     <style>''',
            #     unsafe_allow_html=True)
            container_summary = st.container(border=True)
            num_pred_match = 0
            match_tab, mismatch_tab = st.tabs(["Matched Predictions", "Mismatched Predictions"])
            for uploaded_image in uploaded_images:
                img, img_arr, orig = image_processing_function(uploaded_image, input_img_dims)
                predictions = model.predict(img_arr)
                predicted_label_index = np.argmax(predictions, axis=1)[0]
                predicted_label = view_labels.get(prod_cat)[predicted_label_index]
                if predicted_label == prod_cat_class:
                    num_pred_match += 1
                    container_match = match_tab.container(border=True)
                    left_column, middle_column, right_column = container_match.columns([1,1,1])
                else:
                    container_mismatch = mismatch_tab.container(border=True)
                    left_column, middle_column, right_column = container_mismatch.columns([1,1,1])

                left_column.image(img)

                gc = gcam.GradCAM(model=model, classIdx=predicted_label_index)
                heatmap = gc.compute_heatmap(img_arr, verbose=True)
                heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]),
                                    interpolation=cv2.INTER_CUBIC)
                (heatmap, output) = gc.overlay_heatmap(heatmap, orig, alpha=0.45)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

                middle_column.image(output)

                df = pd.DataFrame({"probs": predictions[0]}).sort_values(by="probs", ascending=False).reset_index()
                right_column.markdown(f'**Predicted View = {predicted_label}**\n\n'
                                    'View propabilities: \n\n'
                                    f'- {view_labels.get(prod_cat)[df["index"][0]]} = {round(df["probs"][0]*100, 2)}%\n\n'
                                    f'- {view_labels.get(prod_cat)[df["index"][1]]} = {round(df["probs"][1]*100, 2)}%\n\n'
                                    f'- {view_labels.get(prod_cat)[df["index"][2]]} = {round(df["probs"][2]*100, 2)}%\n\n'
                                    f'- {view_labels.get(prod_cat)[df["index"][3]]} = {round(df["probs"][3]*100, 2)}%\n\n'
                                    f'- {view_labels.get(prod_cat)[df["index"][4]]} = {round(df["probs"][4]*100, 2)}%\n\n'
                                    )
            accuracy = round(((num_pred_match / len(uploaded_images))*100), 2)
            container_summary.markdown(f"<p style='font-size:18px;'><b>{num_pred_match}</b> predictions out of a total of <b>{len(uploaded_images)}</b> images match the selected product view. Accurcy = <b>{accuracy}%</b></p>", unsafe_allow_html=True)