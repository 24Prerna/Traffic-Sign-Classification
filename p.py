import tensorflow as tf
model = tf.keras.models.load_model("traffic_sign_model.keras")
print("Model input shape:", model.input_shape)
