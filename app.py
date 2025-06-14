import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import os

# PyTorch 2.6 compatibility fix
original_torch_load = torch.load


def safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)


torch.load = safe_torch_load

# TACO dataset classes
CLASS_NAMES = [
    'Bottle', 'Bottle cap', 'Can', 'Cigarette', 'Cup',
    'Lid', 'Other', 'Plastic bag + wrapper', 'Pop tab', 'Straw'
]

# Class colors for visualization
CLASS_COLORS = {
    'Bottle': (59, 130, 246),
    'Bottle cap': (239, 68, 68),
    'Can': (16, 185, 129),
    'Cigarette': (245, 158, 11),
    'Cup': (139, 92, 246),
    'Lid': (236, 72, 153),
    'Other': (107, 114, 128),
    'Plastic bag + wrapper': (20, 184, 166),
    'Pop tab': (249, 115, 22),
    'Straw': (132, 204, 22)
}


@st.cache_resource
def load_model():
    """Load the YOLO model"""
    try:
        # Try to find the model file
        model_path = "taco_model.pt"  # Update this path to your model

        if not os.path.exists(model_path):
            # Look for any .pt files in current directory
            pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
            if pt_files:
                model_path = pt_files[0]
            else:
                st.error("❌ No model file found! Please add your YOLO model (.pt file) to the app directory.")
                return None

        model = YOLO(model_path)
        st.success(f"✅ Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


def draw_predictions(image, results):
    """Draw bounding boxes and labels on the image"""
    img_with_boxes = image.copy()
    img_array = np.array(img_with_boxes)

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())

                if class_id >= len(CLASS_NAMES):
                    continue

                class_name = CLASS_NAMES[class_id]
                color = CLASS_COLORS.get(class_name, (107, 114, 128))

                # Draw bounding box
                cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                # Draw label background
                cv2.rectangle(img_array, (x1, y1 - text_height - 10),
                              (x1 + text_width, y1), color, -1)

                # Draw label text
                cv2.putText(img_array, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return Image.fromarray(img_array)


def main():
    # Page config
    st.set_page_config(
        page_title="WasteNet AI",
        page_icon="🗑️",
        layout="wide"
    )

    # Header
    st.title("🗑️ WasteNet - AI Waste Detection")
    st.markdown("Upload an image to detect and classify waste objects using AI")

    # Load model
    model = load_model()

    if model is None:
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'webp'],
        help="Upload an image containing waste objects"
    )

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_column_width=True)

        # Run inference
        with st.spinner("🤖 AI is analyzing the image..."):
            try:
                # Convert PIL to OpenCV format
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Run YOLO inference
                results = model(cv_image)

                # Draw predictions
                result_image = draw_predictions(image, results)

                # Count detections
                total_detections = 0
                detections_by_class = {}

                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())

                            if class_id < len(CLASS_NAMES):
                                class_name = CLASS_NAMES[class_id]
                                total_detections += 1

                                if class_name not in detections_by_class:
                                    detections_by_class[class_name] = []
                                detections_by_class[class_name].append(confidence)

                with col2:
                    st.subheader("🎯 Detection Results")
                    st.image(result_image, use_column_width=True)

                # Results summary
                st.markdown("---")
                st.subheader("📊 Detection Summary")

                if total_detections > 0:
                    # Metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Objects", total_detections)

                    with col2:
                        avg_confidence = np.mean([conf for confs in detections_by_class.values() for conf in confs])
                        st.metric("Avg Confidence", f"{avg_confidence:.1%}")

                    with col3:
                        st.metric("Object Types", len(detections_by_class))

                    # Detailed results
                    st.subheader("🏷️ Detected Objects")

                    for class_name, confidences in detections_by_class.items():
                        count = len(confidences)
                        max_conf = max(confidences)
                        avg_conf = np.mean(confidences)

                        st.write(f"**{class_name}**: {count} object(s)")
                        st.write(f"   • Best confidence: {max_conf:.1%}")
                        st.write(f"   • Average confidence: {avg_conf:.1%}")
                        st.write("")

                else:
                    st.info("🔍 No waste objects detected in this image")

            except Exception as e:
                st.error(f"❌ Error during inference: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using **Streamlit** and **YOLOv8** | "
        "Dataset: [TACO (Trash Annotations in Context)](http://tacodataset.org/)"
    )


if __name__ == "__main__":
    main()