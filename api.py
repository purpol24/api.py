from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import mysql.connector
import os
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Custom layer definition
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

# Get the base directory for models
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Load the trained model and tokenizer
model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "snn_classifier_trained_without_lambda.keras"),
    custom_objects={'AbsDiffLayer': AbsDiffLayer},
    compile=False
)

tokenizer = pickle.load(open(os.path.join(MODELS_DIR, "tokenizer.pickle"), "rb"))
print("Tokenizer vocabulary:", tokenizer.word_index)
sys.stdout.flush()

encoder = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "encoder_model.keras"),
    compile=False
)

MAX_LEN = 10
THRESHOLD_SUGGEST = 0.40  

# Synonym expansion for better matching
SYNONYMS = {
    "web development":      ["html", "css", "javascript", "react", "vue", "tailwind", "front-end development", "full-stack development", "web design"],
    "ai & machine learning": ["artificial intelligence", "deep learning", "neural networks", "natural language processing", "data modeling", "supervised learning", "unsupervised learning"],
    "cybersecurity & ethical hacking": ["information security", "ethical hacking", "penetration testing", "cyber defense", "network security", "vulnerability assessment", "malware analysis", "red teaming", "white hat hacking", "penetration testing", "network penetration", "hacker tools", "social engineering", "wireshark", "nmap", "kali linux", "reverse engineering"],
    "ui/ux design & product design": ["user interface design", "user experience design", "interaction design", "visual design", "product design", "mobile design", "wireframing", "prototyping", "css"],
    "devops & it infrastructure": ["devops engineering", "cloud infrastructure", "continuous integration", "continuous delivery", "infrastructure as code", "kubernetes", "docker", "jenkins", "cloud automation", "server management", "ansible", "terraform", "git", "ci/cd"],
    "data science & big data analytics": ["data analysis", "data analytics", "data mining", "big data engineering", "data visualization", "statistical analysis", "machine learning", "predictive modeling", "hadoop", "apache spark", "pandas", "matplotlib", "numpy", "r programming", "sql"],
    "software engineering & development": ["software development", "application development", "system architecture", "backend development", "object-oriented programming", "java development", "c++ development", "software engineering practices", "code optimization", "agile software development"],
    "cloud computing":       ["cloud architecture", "cloud solutions", "cloud services", "cloud systems", "virtualization", "cloud deployment", "serverless architecture", "cloud infrastructure", "cloud security", "aws", "azure", "google cloud", "cloud services management"],
    "mobile app development": ["mobile development", "app development", "iOS development", "android development", "mobile software development", "react native development", "flutter development", "mobile ui/ux", "native app development", "xamarin", "swift", "kotlin"],
    "product management & agile development": ["product strategy", "product roadmapping", "agile project management", "scrum management", "agile methodologies", "lean product development", "product lifecycle", "scrum master", "product vision"],
    "game development":       ["game design", "game programming", "video game development", "unity development", "unreal engine", "game mechanics", "game animation", "3d modeling for games", "mobile game development"],
    "network administration": ["network management", "network engineering", "IT networks", "network configuration", "systems administration", "lan/wan management", "routing and switching", "network security administration", "vpn configuration", "network troubleshooting", "ip addressing", "firewall configuration", "dns management", "tcp/ip", "cisco", "router configuration", "switch configuration", "subnetting", "dhcp", "ipv4", "ipv6"],
    "it support":             ["it assistance", "technical support", "help desk support", "systems support", "it troubleshooting", "hardware and software support", "end-user support", "desktop support", "it service management", "technical assistance"],
    "data engineering":       ["data pipeline engineering", "etl processes", "data architecture", "data warehousing", "big data engineering", "sql database management", "nosql databases", "data integration", "hadoop engineering", "real-time data processing", "spark", "kafka"],
    "resume building":        ["resume", "cv", "portfolio", "job application", "cover letter"],
    "leadership & team management": ["leadership", "team management", "project management", "teamwork", "communication", "conflict resolution"],
    "interview preparation":  ["interview", "mock interview", "job interview", "interview questions", "interview skills", "behavioral interview"],

    # Networking fieldwork & cabling synonyms
    "structured cabling": [
        "cabling", "cable management", "patch panel", "keystone jack", "punch down", "punchdown tool",
        "crimping", "crimping tool", "rj45 termination", "cable termination", "ethernet cabling",
        "cat5e", "cat6", "cat6a", "fiber optic cabling", "fiber termination", "splicing",
        "otdr testing", "certification testing", "fluke testing"
    ],
    "cabling": [
        "structured cabling", "cable management", "patch panel", "keystone jack", "punch down",
        "punchdown tool", "crimping", "crimping tool", "rj45 termination", "cable termination",
        "ethernet cabling"
    ],
    "cable management": [
        "structured cabling", "cabling", "rack cable management", "patch panel", "keystone jack",
        "punchdown tool", "crimping tool"
    ]
}

def expand_synonyms(skills):
    expanded = []
    for skill in skills:
        skill_lower = skill.lower()
        if skill_lower in SYNONYMS:
            # Preserve multi-word phrases instead of splitting into separate tokens
            expanded.extend([syn.lower() for syn in SYNONYMS[skill_lower]])
        else:
            # Keep the original skill phrase intact
            expanded.append(skill_lower)
    return expanded

# Function to encode text (skills)
def encode_texts(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN)

# Cosine similarity function
def overall_cos(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        print("Zero vector encountered in cosine similarity calculation.")
        return 0.0
    a_n = a / np.linalg.norm(a)
    b_n = b / np.linalg.norm(b)
    return float(np.dot(a_n, b_n))

def get_skill_embedding(skill, tokenizer, embedding_matrix):
    tokens = skill.lower().split()
    indices = [tokenizer.word_index.get(token) for token in tokens if token in tokenizer.word_index]
    if not indices:
        return np.zeros(embedding_matrix.shape[1])
    vectors = [embedding_matrix[idx] for idx in indices]
    return np.mean(vectors, axis=0)

# ——— MATCH MENTORS ———
@app.route('/match_mentors', methods=['POST'])
def match_mentors():
    try:
        data = request.json  
        if not data:
            return jsonify({"error": "No data provided"}), 400

        mentee_id = data.get('mentee_id')
        mentee_skills = [skill.strip().lower() for skill in data.get('skills', [])]

        if not mentee_skills:
            return jsonify({"error": "No skills provided"}), 400

        # Get pagination parameters
        limit = int(data.get('limit', 3))  # Default to 3 results per page
        offset = int(data.get('offset', 0))  # Default to start from beginning  

        # Expand synonyms for better matches
        expanded_skills = expand_synonyms(mentee_skills)
        print("Expanded skills:", expanded_skills)

        # Get embedding matrix from encoder's first layer
        embedding_matrix = encoder.layers[0].get_weights()[0]

        # Compute mentee mean embedding using get_skill_embedding
        mentee_embs = [get_skill_embedding(skill, tokenizer, embedding_matrix) for skill in expanded_skills]
        mentee_mean = np.mean(mentee_embs, axis=0)
        print("Mentee mean embedding:", mentee_mean)

        # Connect to the MySQL database
        db_connection = mysql.connector.connect(
            host=os.environ.get("MYSQLHOST", "localhost"),
            user=os.environ.get("MYSQLUSER", "root"),
            password=os.environ.get("MYSQLPASSWORD", ""),
            database=os.environ.get("MYSQLDATABASE", "gabaykareradb"),
            port=int(os.environ.get("MYSQLPORT", 3306))
        )

        cursor = db_connection.cursor(dictionary=True)
        cursor.execute("SELECT id, first_name, last_name, availability, profile_picture, job_title FROM user_tb WHERE role_id = 2")
        mentors = cursor.fetchall()

        if not mentors:
            return jsonify({"error": "No mentors found in the database"}), 400
    
        results = []
        for mentor in mentors:
            mentor_id = mentor['id']
            
            # Check if mentor is fully booked (has 2 or more mentees with pending/ongoing status)
            cursor.execute("""
                SELECT COUNT(DISTINCT mentee_id) as active_mentee_count 
                FROM sessions 
                WHERE mentor_id = %s 
                AND status IN ('pending', 'ongoing')
            """, (mentor_id,))
            booking_status = cursor.fetchone()
            active_mentee_count = booking_status['active_mentee_count'] if booking_status else 0
            
            # Skip mentor if they already have 2 or more active mentees
            if active_mentee_count >= 2:
                continue
            
            cursor.execute("SELECT skill_name FROM mentor_skills_tb WHERE mentor_id = %s", (mentor_id,))
            skills = cursor.fetchall()

            if not skills:
                continue

            mentor_skills = []
            for skill in skills:
                mentor_skills.extend([s.strip().lower() for s in skill['skill_name'].split(',')])

            # Compute mentor mean embedding using get_skill_embedding
            mentor_embs = [get_skill_embedding(skill, tokenizer, embedding_matrix) for skill in mentor_skills]
            mentor_mean = np.mean(mentor_embs, axis=0)

            # Calculate similarity using the direct difference computation function
            sim = overall_cos(mentee_mean, mentor_mean)

            # Normalize score to [0, 1]
            score = (sim + 1.0) / 2.0

            # Check for exact matches
            exact_matches = [f"{ms}→{ts}(1.00)" for ms in expanded_skills for ts in mentor_skills if ms == ts]
            suggestions = [f"{ms}→{ts}({sim:.2f})" for ms in expanded_skills for ts in mentor_skills if ms != ts and sim >= THRESHOLD_SUGGEST]

            # Only add mentors with valid matches or suggestions
            if exact_matches or suggestions:
                results.append({
                    "id": mentor['id'],
                    "name": f"{mentor['first_name']} {mentor['last_name']}",
                    "job_title": mentor.get('job_title'),  # Include job_title from database
                    "score": score,
                    "skills": mentor_skills,
                    "availability": mentor['availability'],
                    "profile_picture": mentor['profile_picture'],
                    "exact_matches": exact_matches,  
                    "suggestions": suggestions  
                })

        # Sort results by similarity score (highest to lowest)
        results.sort(key=lambda x: x['score'], reverse=True)

        # Apply pagination: return results based on offset and limit
        paginated_results = results[offset:offset + limit]
        
        # Return results with metadata about whether there are more results
        response_data = {
            "results": paginated_results,
            "has_more": len(results) > offset + limit,
            "total": len(results),
            "offset": offset,
            "limit": limit
        }

        # Close database connection
        cursor.close()
        db_connection.close()

        return jsonify(response_data)

    except Exception as e:
        # Log the exception for better debugging
        app.logger.error(f"Error: {str(e)}")
        # Close database connection if it exists
        try:
            if 'cursor' in locals():
                cursor.close()
            if 'db_connection' in locals():
                db_connection.close()
        except:
            pass
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
