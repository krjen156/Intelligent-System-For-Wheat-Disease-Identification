import os
import time
import numpy as np
import streamlit as st
from PIL import Image

from config import MODELS
from model_utils import load_keras_model, prepare_image, get_input_size
from gradcam_utils import make_gradcam_heatmap, overlay_gradcam

st.set_page_config(
    page_title="Wheat Disease Detection",
    layout="wide"
)

st.title("🌾 Intelligent System for Wheat Disease Identification")

tab_howto, tab_predict, tab_models = st.tabs(["❓ About","🔍 Prediction", "📊 Model Information"])

@st.cache_resource
def cached_load_model(model_path):
    return load_keras_model(model_path)

with tab_howto:
    st.header("About the Intelligent System for Wheat Disease Identification")

    st.subheader("General knowledge")
    st.write("""
     The system is designed to provide a simple and intuitive way to identify wheat diseases from images. 
     To use the system, the user uploads an image of a wheat plant through the interface and selects one or more trained models for evaluation. 
     The system will then process the image, run it through the selected models, and return predictions including the detected disease class, confidence score, and a Grad-CAM visualization highlighting the regions of the image that influenced the prediction. 
     By selecting multiple models, users can compare results and gain a more reliable understanding of the diagnosis.
     """)

    st.subheader("Limitations")
    st.write("""
    The system is specifically trained on wheat plant datasets and should only be used with images of wheat plants. 
    Images of other crops, plants, or unrelated objects may lead to incorrect predictions, as the models are not trained to recognize anything outside the wheat domain. 
    Additionally, the system does not verify whether the uploaded image actually contains a wheat plant. 
    If there is uncertainty about whether the image depicts wheat, the system should not be relied upon for disease detection.
    """)

    st.subheader("Study")
    st.write("""
    The intelligent system is based on a comprehensive research study conducted as part of this project. 
    The original paper describing the system can be downloaded from the provided link within the application. 
    The paper offers a detailed overview of the entire development process, including related work, methodology, and experimental results.
    It thoroughly explains how each model was designed, trained, and evaluated, and provides insights into the comparative performance across different datasets.
    By reviewing the paper, users can gain a deeper understanding of the technical foundation behind the system, as well as the reasoning behind model selection and implementation choices.
    """)

with tab_predict:
    st.header("Upload and Compare Models")

    uploaded_file = st.file_uploader(
        "Upload wheat image",
        type=["jpg", "jpeg", "png"]
    )
    selected_models = st.multiselect(
        "Select models",
        options=list(MODELS.keys()),
        default=["ResNet50 - WPD", "VGG16 - WPD"]
    )
    if uploaded_file and selected_models:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("Uploaded Image")
        st.image(image, width=350)
        st.divider()
        for model_name in selected_models:
            st.header(model_name)
            config = MODELS[model_name]
            if not os.path.exists(config["path"]):
                st.error(f"Model not found: {config['path']}")
                continue
            total_start = time.time()
            model = cached_load_model(config["path"])
            input_size = get_input_size(model)
            pre_start = time.time()
            img_array = prepare_image(
                image=image,
                input_size=input_size,
                preprocess_function=config["preprocess"]
            )
            pre_time = time.time() - pre_start
            _ = model(img_array, training=False)
            pred_start = time.time()
            predictions = model.predict(img_array, verbose=0)
            pred_time = time.time() - pred_start

            if isinstance(predictions, list):
                predictions = predictions[0]
            predicted_index = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][predicted_index])
            class_names = config["classes"]
            if predicted_index >= len(class_names):
                st.error("Class mismatch")
                continue
            predicted_class = class_names[predicted_index]
            cam_start = time.time()
            heatmap = make_gradcam_heatmap(
                img_array,
                model,
                predicted_index
            )
            cam_time = time.time() - cam_start
            total_time = time.time() - total_start
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("### Prediction")
                st.write(f"**Class:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2%}")
                train_acc = config.get("training accuracy")
                eval_acc = config.get("final evaluation accuracy")
                if train_acc:
                    st.write(f"**Training accuracy:** {train_acc:.2%}")
                else:
                    st.write("**Training accuracy:** N/A")
                if eval_acc:
                    st.write(f"**Evaluation accuracy:** {eval_acc:.2%}")
                else:
                    st.write("**Evaluation accuracy:** N/A")
                st.write(f"**Runtime:** {total_time:.2f}s")
            with col2:
                st.markdown("### Original")
                st.image(image, use_container_width=True)
            with col3:
                st.markdown("### Grad-CAM")
                if heatmap is not None:
                    gradcam_img = overlay_gradcam(
                        original_image=image,
                        heatmap=heatmap,
                        input_size=input_size
                    )
                    st.image(gradcam_img, use_container_width=True)
                else:
                    st.warning("Grad-CAM not available")
            probabilities = {
                class_names[i]: float(predictions[0][i])
                for i in range(min(len(class_names), len(predictions[0])))
            }
            st.markdown("### Class Probabilities")
            st.json({
                "model": model_name,
                "prediction": predicted_class,
                "confidence": confidence,
                "training_accuracy": train_acc,
                "evaluation_accuracy": eval_acc,
                "probabilities": probabilities,
                "timing": {
                    "preprocessing": round(pre_time, 3),
                    "prediction": round(pred_time, 3),
                    "gradcam": round(cam_time, 3),
                    "total": round(total_time, 3)
                }
            })

            st.divider()


with tab_models:
    st.header("Model Descriptions")

    st.subheader("ResNet50")
    st.write("""
    ResNet50 is a deep convolutional neural network that uses residual connections 
    to allow training of very deep architectures. These skip connections help prevent 
    vanishing gradients and improve feature learning. In this project, ResNet50 showed strong generalization performance, especially 
    on more complex datasets with multiple disease classes.
    """)

    st.subheader("VGG16")
    st.write("""
    VGG16 is a classical CNN architecture based on stacked 3x3 convolutional layers. 
    It is deeper and more computationally expensive than simpler models, but it produces 
    stable and reliable predictions. VGG16 performed consistently across datasets and showed strong classification ability.
    """)

    st.subheader("WheatNetwork (Custom CNN)")
    st.write("""
    WheatNetwork is a custom-designed convolutional neural network developed specifically 
    for this project. It focuses on learning disease-specific visual patterns such as 
    discoloration, texture, and shape variations. Despite being smaller than pretrained models, it performed competitively, especially 
    on datasets with more classes.
    """)

    st.header("Wheat Diseases")
    st.image("Images/disease types2.jpg", caption="Grouping and comparing dataset classes (diseases) with potentially yield losses and disease type")

