import tensorflow as tf
import streamlit as st
import numpy as np

# Load the model
model = tf.keras.models.load_model("Fruit_classifer.keras")

# Define image dimensions
img_h = 480
img_w = 480

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = tf.image.decode_image(image.read(), channels=3)
    img = tf.image.resize(img, [img_h, img_w])
    img = tf.expand_dims(img, axis=0)
    return img

# Set page config
st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍏",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS styles
st.markdown(
    """
    <style>
    .st-ba {
        background-color: #f0f2f6;
    }
    .st-cy {
        color: #0078FF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display file uploader and get user uploaded image
st.title("🍏 Fruit Classifier")
img_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if img_file is not None:
    img = preprocess_image(img_file)

    # Make predictions
    predictions = model.predict(img)

    # List of fruit names
    c_names = ['Acerolas', 'Apples', 'Apricots', 'Avocados', 'Bananas', 'Blackberries', 'Blueberries',
               'Cantaloupes', 'Cherries', 'Coconuts', 'Figs', 'Grapefruits', 'Grapes', 'Guava', 'Kiwifruit',
               'Lemons', 'Limes', 'Mangos', 'Olives', 'Oranges', 'Passionfruit', 'Peaches', 'Pears', 'Pineapples',
               'Plums', 'Pomegranates', 'Raspberries', 'Strawberries', 'Tomatoes', 'Watermelons']

    # Get top 3 predicted classes
    predicted_class = tf.math.top_k(predictions, k=3)
    class_names = [c_names[idx] for idx in predicted_class.indices[0]]
    accuracy = predicted_class.values[0].numpy() * 100

    # Display results
    st.subheader("Uploaded Image")
    img_np = img.numpy()[0].astype(np.uint8)  # Convert EagerTensor to NumPy array
    st.image(img_np, caption="Uploaded Image", use_column_width=True)

    st.subheader("Prediction Results")
    for i, (class_name, acc) in enumerate(zip(class_names, accuracy)):
        st.write(
            f"Prediction {i + 1}: **{class_name}** with accuracy of **{acc:.2f}%**",
        )
