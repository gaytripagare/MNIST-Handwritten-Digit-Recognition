import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# 2. Model Loading
# -----------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("mnist_cnn_improved.keras")


try:
    model = load_cnn_model()
except Exception as e:
    st.error("⚠️ Could not load `mnist_cnn.keras`. Make sure the model file exists in your working directory.")
    st.stop()

# -----------------------------
# 3. Sidebar Controls
# -----------------------------
st.sidebar.title("🛠️ Drawing Tools & Settings")
st.sidebar.markdown("Customize your canvas and drawing style.")

stroke_width = st.sidebar.slider("Brush Thickness", min_value=10, max_value=30, value=20, step=2)
drawing_mode = st.sidebar.radio("Tool", options=["freedraw", "transform"], index=0,
                                format_func=lambda x: "✏️ Draw" if x == "freedraw" else "✋ Move/Transform")

st.sidebar.divider()
st.sidebar.markdown("### 💡 Tips for Best Accuracy")
st.sidebar.info(
    """
    - **Center your digit** in the middle of the canvas.
    - Draw with **bold, continuous lines**.
    - Fill reasonable space, similar to MNIST training samples.
    """
)

# -----------------------------
# 4. Main Header
# -----------------------------
st.title("🔢 Interactive Handwritten Digit Recognition")
st.markdown("Draw a digit from **0 to 9** on the canvas below and see the neural network analyze it in real time!")

st.divider()

# -----------------------------
# 5. Canvas & Output Layout
# -----------------------------
col_canvas, col_results = st.columns([1, 1], gap="large")

with col_canvas:
    st.subheader("✍️ Canvas")

    # Drawable Canvas Component
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=stroke_width,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode=drawing_mode,
        key="digit_canvas",
    )

    predict_btn = st.button("🔮 Predict Digit", type="primary", use_container_width=True)

with col_results:
    st.subheader("📊 Model Insights")

    # Run prediction if canvas has drawing and button is clicked
    if canvas_result.image_data is not None and predict_btn:

        # Check if user actually drew something (non-black pixels)
        raw_img = canvas_result.image_data
        if np.sum(raw_img[:, :, :3]) == 0:
            st.warning("⚠️ Canvas is empty! Please draw a digit first.")
        else:
            with st.spinner("Analyzing drawing..."):
                # 1. Image Preprocessing
                img = Image.fromarray(raw_img.astype("uint8")).convert("L")
                img_resized = img.resize((28, 28))
                img_array = np.array(img_resized).astype("float32") / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)

                # 2. Model Prediction
                prediction = model.predict(img_array, verbose=0)[0]
                predicted_digit = int(np.argmax(prediction))
                confidence = float(np.max(prediction) * 100)

            # High confidence celebration!
            if confidence > 85.0:
                st.balloons()

            # Display Predictions
            st.markdown(f"### 🎯 Predicted Digit: **{predicted_digit}**")

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Confidence Score", value=f"{confidence:.1f}%")
            with res_col2:
                # Show what the model actually sees (28x28)
                st.caption("Model Input View (28x28):")
                st.image(img_resized, width=80)

            # Interactive Probability Chart
            st.markdown("#### Probability Distribution")
            df_prob = pd.DataFrame({
                "Digit": [str(i) for i in range(10)],
                "Probability (%)": [p * 100 for p in prediction]
            }).set_index("Digit")

            st.bar_chart(df_prob, height=200)

            # Interactive Feedback Section
            st.divider()
            st.write("##### Was this prediction correct?")
            fb_col1, fb_col2 = st.columns(2)
            if fb_col1.button("👍 Yes", key="yes_fb"):
                st.toast("Awesome! Glad the CNN got it right 🎉")
            if fb_col2.button("👎 No", key="no_fb"):
                st.toast("Thanks for the feedback! Try centering or drawing thicker lines.")

    elif not predict_btn:
        st.info("👈 Draw a digit on the left and click **Predict Digit** to get results.")

# -----------------------------
# 6. Deep Dive / Explainer Section
# -----------------------------
st.divider()
with st.expander("🔍 How does this model work under the hood?"):
    st.markdown(
        """
        1. **Grayscale Conversion**: Your RGB canvas drawing is converted into a single-channel grayscale image.
        2. **Downsampling to 28x28**: The 280x280 pixel canvas is scaled down to a standard **28x28 pixel grid** matching the standard MNIST dataset layout.
        3. **Normalization**: Pixel intensities (0 to 255) are divided by 255 to scale values between `0.0` and `1.0`.
        4. **Softmax Output**: The Convolutional Neural Network (CNN) produces a probability vector over 10 output nodes (digits 0–9).
        """
    )