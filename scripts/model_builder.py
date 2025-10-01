from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable

# Registering the Lambda function for serialization
@register_keras_serializable()
def abs_diff(x):
    return tf.abs(x[0] - x[1])

# Model parameters
MAX_LEN = 10
EMBED_DIM = 64
vocab_size = 1000  # Adjust this based on your tokenizer vocabulary size

def build_encoder():
    inp = layers.Input(shape=(MAX_LEN,), name="skill_input")
    x   = layers.Embedding(vocab_size, EMBED_DIM, name="embed")(inp)
    x   = layers.LSTM(64, name="lstm")(x)
    x   = layers.Dense(128, activation="relu", name="dense1")(x)
    x   = layers.Dropout(0.3, name="dropout")(x)
    x   = layers.Dense(64, activation="relu", name="dense2")(x)
    return models.Model(inp, x, name="encoder")

def build_model():
    encoder = build_encoder()

    # Define inputs
    left = layers.Input(shape=(MAX_LEN,), name="left_input")
    right = layers.Input(shape=(MAX_LEN,), name="right_input")

    # Pass inputs through the encoder
    va = encoder(left)
    vb = encoder(right)

    # Lambda layer with the registered abs_diff function
    diff = layers.Lambda(abs_diff, output_shape=(64,))([va, vb])  # Explicit output shape
    mul = layers.Multiply(name="mul")([va, vb])
    concat = layers.Concatenate(name="concat")([diff, mul])

    # Fully connected layers after concatenation
    h = layers.Dense(64, activation="relu", name="head")(concat)
    out = layers.Dense(1, activation="sigmoid", name="match_prob")(h)

    # Build and compile the model
    model = models.Model([left, right], out, name="siamese_classifier")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(model.summary())

    return model

# Build the model
model = build_model()

# Save the model and encoder
model_path = "C:/PROJECTS/GabayKarera/storage/app/models/snn_classifier.keras"
encoder_path = "C:/PROJECTS/GabayKarera/storage/app/models/encoder_model.keras"

model.save(model_path)  # Save the main model
encoder = model.get_layer("encoder")  # Extract the encoder part
encoder.save(encoder_path)  # Save the encoder model

print(f"✅ Model saved to {model_path}")
print(f"✅ Encoder saved to {encoder_path}")
