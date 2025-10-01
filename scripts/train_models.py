import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Constants
MAX_LEN = 10
EMBEDDING_DIM = 100
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

@tf.keras.utils.register_keras_serializable()
class AbsDiffLayer(tf.keras.layers.Layer):
    """Custom layer to compute absolute difference between two tensors"""
    def __init__(self, **kwargs):
        super(AbsDiffLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])

    def get_config(self):
        config = super(AbsDiffLayer, self).get_config()
        return config

def create_encoder_model(vocab_size, max_len):
    """Create the encoder model for skill embeddings"""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=max_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu')
    ])
    return model

def create_snn_model(encoder):
    """Create the Siamese Neural Network model"""
    input1 = tf.keras.layers.Input(shape=(MAX_LEN,))
    input2 = tf.keras.layers.Input(shape=(MAX_LEN,))
    
    # Encode both inputs using the same encoder
    encoded1 = encoder(input1)
    encoded2 = encoder(input2)
    
    # Compute absolute difference using custom layer
    diff = AbsDiffLayer()([encoded1, encoded2])
    
    # Classification layers
    x = tf.keras.layers.Dense(64, activation='relu')(diff)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    return model

def main():
    # Load and preprocess the dataset
    print("Loading dataset...")
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                               "Expanded_Skill_Pairing_Dataset.csv")
    df = pd.read_csv(dataset_path)
    
    # Create and fit tokenizer on both single words and phrases
    print("Creating tokenizer...")
    all_skills = pd.concat([df['skill_mentee'], df['skill_mentor']]).unique()
    # Split all phrases into words and add to the list
    all_words = set()
    for skill in all_skills:
        for word in str(skill).lower().split():
            all_words.add(word)
    # Combine both phrases and single words
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list(all_skills) + list(all_words))
    
    # Save tokenizer
    print("Saving tokenizer...")
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "tokenizer.pickle"), "wb") as f:
        pickle.dump(tokenizer, f)
    
    # Convert skills to sequences
    mentee_seqs = tokenizer.texts_to_sequences(df['skill_mentee'])
    mentor_seqs = tokenizer.texts_to_sequences(df['skill_mentor'])
    
    # Pad sequences
    mentee_padded = pad_sequences(mentee_seqs, maxlen=MAX_LEN)
    mentor_padded = pad_sequences(mentor_seqs, maxlen=MAX_LEN)
    
    # Create and train encoder
    print("Training encoder model...")
    vocab_size = len(tokenizer.word_index) + 1
    encoder = create_encoder_model(vocab_size, MAX_LEN)
    encoder.compile(optimizer='adam', loss='mse')
    
    # Train encoder on both mentee and mentor skills
    encoder.fit(
        np.concatenate([mentee_padded, mentor_padded]),
        np.zeros((len(mentee_padded) + len(mentor_padded), 32)),  # Dummy target
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )
    
    # Save encoder
    print("Saving encoder model...")
    encoder.save(os.path.join(models_dir, "encoder_model.keras"))
    
    # Create and train Siamese Neural Network
    print("Training Siamese Neural Network...")
    snn_model = create_snn_model(encoder)
    snn_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train SNN
    snn_model.fit(
        [mentee_padded, mentor_padded],
        df['label'].values,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1
    )
    
    # Save SNN model
    print("Saving SNN model...")
    snn_model.save(os.path.join(models_dir, "snn_classifier_trained_without_lambda.keras"))
    
    print("Training complete! Models saved in models/ directory.")

if __name__ == "__main__":
    main() 