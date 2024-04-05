import streamlit as st
import numpy as np
import pandas as pd
import cv2
import boto3
import gradcam as gcam # Helper file contains the class definition for GradCAM

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils

S3_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
S3_SECRET_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
S3_REGION = st.secrets["AWS_DEFAULT_REGION"]

st.set_page_config(page_title="Image View Classifier", layout="wide")
client_s3 = boto3.client("s3", region_name=S3_REGION, aws_access_key_id=S3_KEY_ID, aws_secret_access_key=S3_SECRET_KEY)

@st.cache_resource
def load_models():
    models = {}
    result = client_s3.download_file("bestsecret-models",'bag_resnet50_model_ft_all_93pct.h5', "/tmp/bag_resnet50_model_ft_all_93pct.h5")
    result = client_s3.download_file("bestsecret-models",'clothes_resnet50_func_model_97pct.h5', "/tmp/clothes_resnet50_func_model_97pct.h5")
    result = client_s3.download_file("bestsecret-models",'schuhe_resnet50_model_ft_all_94pct.h5', "/tmp/schuhe_resnet50_model_ft_all_94pct.h5")
    result = client_s3.download_file("bestsecret-models",'waesch_funcResnet_model_94pct.h5', "/tmp/waesch_funcResnet_model_94pct.h5")

    model_bag = load_model("/tmp/bag_resnet50_model_ft_all_93pct.h5", custom_objects={'imagenet_utils': imagenet_utils})
    model_clothes = load_model('/tmp/clothes_resnet50_func_model_97pct.h5', custom_objects={'imagenet_utils': imagenet_utils})
    model_schuhe = load_model('/tmp/schuhe_resnet50_model_ft_all_94pct.h5', custom_objects={'imagenet_utils': imagenet_utils})
    model_waesche = load_model('/tmp/waesch_funcResnet_model_94pct.h5', custom_objects={'imagenet_utils': imagenet_utils})

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
            container_summary.markdown(f"<p style='font-size:18px;'><b>{num_pred_match}</b> predictions out of a total of <b>{len(uploaded_images)}</b> images match the selected product view. Accuracy = <b>{accuracy}%</b></p>", unsafe_allow_html=True)
