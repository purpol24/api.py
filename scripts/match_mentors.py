import sys
import json
import pickle
import numpy as np
import tensorflow as tf
import mysql.connector
from tensorflow.keras.preprocessing.sequence import pad_sequences

# CONFIGURATION
TOKENIZER_PATH = r'C:\PROJECTS\GabayKarera\storage\app\models\tokenizer.pickle'
ENCODER_PATH = r'C:\PROJECTS\GabayKarera\storage\app\models\snn_classifier.keras'
MAX_LEN = 10

# Load the tokenizer and model
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
encoder = tf.keras.models.load_model(ENCODER_PATH, compile=False, safe_mode=False)

# Function to encode texts (skills)
def encode_texts(texts, tokenizer):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN)

# Function to calculate cosine similarity
def overall_cos(a, b):
    # Avoid division by zero by checking for empty vectors
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    a_n = a / np.linalg.norm(a)
    b_n = b / np.linalg.norm(b)
    return float(np.dot(a_n, b_n))

# Input: Mentee skills passed from Laravel or command line (with fallback if not provided)
input_skills = sys.argv[1].split(',') if len(sys.argv) > 1 else input("Enter mentee skills (comma-separated): ").split(',')

# Connect to the MySQL database
db_connection = mysql.connector.connect(
    host="localhost",  # Database host (change if needed)
    user="root",  # Database username
    password="",  # Database password
    database="gabaykareradb"  # Database name
)

# Query to fetch mentors using the role_id
cursor = db_connection.cursor(dictionary=True)
cursor.execute("SELECT id, first_name, last_name FROM user_tb WHERE role_id = 2")  # role_id = 2 for 'mentor'

# Fetch mentor data
mentors = cursor.fetchall()

# Fetch skills for each mentor from mentor_skills_tb
# Fetch skills for each mentor from mentor_skills_tb
# Fetch skills for each mentor from mentor_skills_tb
mentor_data = []
for mentor in mentors:
    mentor_id = mentor['id']
    cursor.execute("SELECT skill_name FROM mentor_skills_tb WHERE mentor_id = %s", (mentor_id,))
    skills = cursor.fetchall()

    # Print out mentor data and skills for debugging
    print(f"Debug: Mentor ID: {mentor_id}, Name: {mentor['first_name']} {mentor['last_name']}, Skills from DB: {skills}")

    # Extract skill names into a list and normalize (lowercase and strip spaces)
    skill_names = [skill['skill_name'].strip().lower() for skill in skills]
    mentor_data.append({
        'name': f"{mentor['first_name']} {mentor['last_name']}",  # Combining first and last name
        'skills': skill_names
    })

# Process mentee skills and normalize (lowercase and strip spaces)
ments = [s.strip().lower() for s in input_skills if s.strip()]

# Check if mentee skills are empty
if not ments:
    print("❌ No mentee skills provided. Exiting.")
    sys.exit()

# Encode mentee skills (only once)
mentee_embs = encode_texts(ments, tokenizer)  # Mentee skills encoded

# Scoring and matching logic
results = []

# Debugging: Print mentee skills
print(f"Debug: Mentee Skills: {ments}")

# Iterate through mentor data and calculate similarity
for mentor in mentor_data:
    mentor_skills = mentor['skills']  # Mentor skills as comma-separated string
    
    # Encode mentee and mentor skills separately
    mentee_embs = encode_texts(ments, tokenizer)  # Mentee skills encoded
    mentor_embs = encode_texts(mentor_skills, tokenizer)  # Mentor skills encoded

    # Calculate similarity score between mentee and mentor
    sim = overall_cos(mentee_embs.mean(axis=0), mentor_embs.mean(axis=0))  # Calculate cosine similarity
    score = (sim + 1.0) / 2.0  # Normalize score to [0, 1]

    # Append mentor match results
    results.append({
        "name": mentor['name'],
        "score": score,
        "skills": mentor['skills'],
    })

# Sort results by similarity score (highest to lowest)
results.sort(key=lambda x: x['score'], reverse=True)

# Select only the top 3 matches
top_3_results = results[:3]

# Return results as JSON (this will be output for Laravel)
print(json.dumps(top_3_results))  # This is returned to Laravel as JSON
