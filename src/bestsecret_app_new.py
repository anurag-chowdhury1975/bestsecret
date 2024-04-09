import streamlit as st
import numpy as np
import pandas as pd
import cv2
import boto3
import gradcam as gcam # Helper file contains the class definition for GradCAM

from PIL import Image
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

input_img_dims = (427, 350)

st.header("Image View Classifier")
# Create an expandable container for the instructions
with st.expander("Instructions on how to use the app...", expanded=False):
    st.markdown("The image classification models in this app have been trained to classify image views associated with 4 specific product categories - Bags, Clothes, Shoes (Schuhe) and Lingerie (Waesche). The images that were used to train the models contained <b>just one product view within each image</b> (not multiple products/views within a single image) on a <b>blank background</b>. In order to get accurate predictions from the model you need to upload images with those characteristics (refer to sample images below).</br></br><b>Step 1:</b> You will need to first select which of the 4 product categories you want the model to predict the views for.<br><b>Step 2:</b> If you want the model to automatically classify and group the views for the images you upload, you can leave the view selection as 'Multiple/All'. If you already already have images grouped by view, then you can select the specific view and cross check what the model predicts. This can help you identify images that you may have grouped incorrectly, or if you are sure you have grouped them correctly, then it will tell you how accuarte the model predictions are.", unsafe_allow_html=True)
    st.markdown("<b>For Bags</b>, here are sample images for the 5 different views that the model expects.", unsafe_allow_html=True)
    st.image('img/bag_views.png', width=700)
    st.markdown("<b>For Clothes</b>, here are sample images for the 5 different views that the model expects.", unsafe_allow_html=True)
    st.image('img/clothes_views.png', width=700)
    st.markdown("<b>For Shoes (Schuhe)</b>, here are sample images for the 5 different views that the model expects.", unsafe_allow_html=True)
    st.image('img/schuhe_views.png', width=700)
    st.markdown("<b>For Lingerie (Waesche)</b>, here are sample images for the 5 different views that the model expects.", unsafe_allow_html=True)
    st.image('img/waesche_views.png', width=700)

prod_cat = st.selectbox(label="Select Product Category", options=list(prod_labels.keys()), format_func=format_func)
model = models.get(prod_cat)
view_label_options = view_labels.get(prod_cat)
prod_cat_class_options = ["Multiple/All"]
prod_cat_class_options.extend(view_label_options)
prod_cat_class = st.selectbox(label="Select Product View", options=prod_cat_class_options)

with st.form("my-form", clear_on_submit=True):
    uploaded_images = st.file_uploader("Please choose image(s)...", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("Predict Image View(s)")
    if submitted:
        if len(uploaded_images) > 0:
            if prod_cat_class == "Multiple/All":
                view_summary = st.container(border=True)
                view0_cnt = view1_cnt = view2_cnt = view3_cnt = view4_cnt = 0
                view_tab0, view_tab1, view_tab2, view_tab3, view_tab4 = st.tabs(view_label_options)

                for uploaded_image in uploaded_images:
                    img, img_arr, orig = image_processing_function(uploaded_image, input_img_dims)
                    predictions = model.predict(img_arr)
                    predicted_label_index = np.argmax(predictions, axis=1)[0]
                    predicted_label = view_labels.get(prod_cat)[predicted_label_index]
                    tab_index = view_label_options.index(predicted_label)
                    if tab_index == 0:
                        view0_cnt += 1
                        container_tab0 = view_tab0.container(border=True)
                        left_column, middle_column, right_column = container_tab0.columns([1,1,1])
                    elif tab_index == 1:
                        view1_cnt += 1
                        container_tab1 = view_tab1.container(border=True)
                        left_column, middle_column, right_column = container_tab1.columns([1,1,1])
                    elif tab_index == 2:
                        view2_cnt += 1
                        container_tab2 = view_tab2.container(border=True)
                        left_column, middle_column, right_column = container_tab2.columns([1,1,1])
                    elif tab_index == 3:
                        view3_cnt += 1
                        container_tab3 = view_tab3.container(border=True)
                        left_column, middle_column, right_column = container_tab3.columns([1,1,1])
                    else:
                        view4_cnt += 1
                        container_tab4 = view_tab4.container(border=True)
                        left_column, middle_column, right_column = container_tab4.columns([1,1,1])

                    # left_column.image(img)
                    left_column.image(orig)

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
                                        f'- {view_label_options[df["index"][0]]} = {round(df["probs"][0]*100, 2)}%\n\n'
                                        f'- {view_label_options[df["index"][1]]} = {round(df["probs"][1]*100, 2)}%\n\n'
                                        f'- {view_label_options[df["index"][2]]} = {round(df["probs"][2]*100, 2)}%\n\n'
                                        f'- {view_label_options[df["index"][3]]} = {round(df["probs"][3]*100, 2)}%\n\n'
                                        f'- {view_label_options[df["index"][4]]} = {round(df["probs"][4]*100, 2)}%\n\n'
                                        )
                view_summary.markdown(f"<p style='font-size:18px;'><b>{len(uploaded_images)}</b> Total Images Uploaded. Views: <b>{view0_cnt}</b> {view_label_options[0]}, <b>{view1_cnt}</b> {view_label_options[1]}, <b>{view2_cnt}</b> {view_label_options[2]}, <b>{view3_cnt}</b> {view_label_options[3]}, <b>{view4_cnt}</b> {view_label_options[4]}</p>", unsafe_allow_html=True)
            else:
                mismatch_summary = st.container(border=True)
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

                    # left_column.image(img)
                    left_column.image(orig)

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
                                        f'- {view_label_options[df["index"][0]]} = {round(df["probs"][0]*100, 2)}%\n\n'
                                        f'- {view_label_options[df["index"][1]]} = {round(df["probs"][1]*100, 2)}%\n\n'
                                        f'- {view_label_options[df["index"][2]]} = {round(df["probs"][2]*100, 2)}%\n\n'
                                        f'- {view_label_options[df["index"][3]]} = {round(df["probs"][3]*100, 2)}%\n\n'
                                        f'- {view_label_options[df["index"][4]]} = {round(df["probs"][4]*100, 2)}%\n\n'
                                        )
                accuracy = round(((num_pred_match / len(uploaded_images))*100), 2)
                mismatch_summary.markdown(f"<p style='font-size:18px;'><b>{num_pred_match}</b> predictions out of a total of <b>{len(uploaded_images)}</b> images match the selected product view. Accuracy = <b>{accuracy}%</b></p>", unsafe_allow_html=True)
