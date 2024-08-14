import tensorflow as tf

# Load the saved model
model_path = '/content/drive/MyDrive/CollegeP5-master/emergencyVehicle.h5'
loaded_model = tf.keras.models.load_model(model_path)

# Define a function to classify input images
def classify_image(image_path):
    # Load and preprocess the input image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = img_array / 255.0  # Normalize pixel values

    # Make predictions
    predictions = loaded_model.predict(img_array)

    # Interpret predictions
    class_label = "Emergency Vehicle" if predictions[0][0] > 0.5 else "Non-Emergency Vehicle"
    confidence = predictions[0][0] if class_label == "Emergency Vehicle" else 1 - predictions[0][0]

    return class_label, confidence

# Example usage
image_path = '/content/drive/MyDrive/CollegeP5-master/Validation/0/n (1039).jpg'
class_label, confidence = classify_image(image_path)
print(f"Class: {class_label}, Confidence: {confidence}")
