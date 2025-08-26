import os
import time
import threading
import datetime
import pickle
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model  # type: ignore
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import nltk
from flask_cors import CORS
# This is needed for tokenization
nltk.download('punkt_tab')

# Get the connection key to the mongo database from the environment variables
# This is set in Sliplane
mongo_uri = os.environ.get("MONGO_URI")

# Globals
# We don't want to initialize these immediately until the server
# is started, so we set them to None and load in a background thread
model = None
tokenizer = None
client = None
db = None
site_visits_col = None
api_queries_col = None


# Start
app = Flask(__name__, static_folder="web/assets")

CORS(app) # Enable CORS for all routes so this can be embedded anywhere, for free

# Cybersecurity: Rate Limiting
# Prevent abuse of the free API
# Adjust the limits as needed
# Here it's set to 100 requests per hour per IP address
limiter = Limiter(
    key_func=get_remote_address,  # how to identify clients
    default_limits=["100 per hour"]  # adjust as needed
)
limiter.init_app(app)

# Helper fn
def sanitizer(tokens):
    return [t.lower() for t in tokens if t.isalnum()]


# Backend Initialization
def load_backend():
    global model, tokenizer, client, db, site_visits_col, api_queries_col
    print("Files in current directory:", os.listdir("."))
    print(os.path.getsize("lstm_model.h5"))
    # Wait for MongoDB
    # The server will be down until it connects to the mongo client
    while True:
        try:
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            client.server_info()
            print("Connected to MongoDB")
            break
        except ServerSelectionTimeoutError as e:
            print("Timeout or cannot reach MongoDB:",e)
            time.sleep(1) 
        except Exception as e:
            print("Error connecting to MongoDB:",e)
            time.sleep(1)

    # Initialize the database and the collections that exist
    # Ifthis is running for the first time, the collections will be created automatically
    db = client["semantic_analysis"]
    site_visits_col = db["site_visits"]
    api_queries_col = db["api_queries"]

    print("Retrieved collections:", db.list_collection_names())

    # Load ML model and tokenizer
    print("Loading ML model and tokenizer...")
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    print("ML Model and tokenizer loaded!")
    print("Backend initialized!")

# Start backend thread
# the api will respond to heartbeats and get this loaded in
threading.Thread(target=load_backend, daemon=True).start()




# -----------------------
# Routes
# -----------------------

# Health Check
# Use this rather than the root so that it will always return 200 if the app is running
@app.route("/health")
def health():
    return "OK", 200



# Before Request Guard
# Use this to ensure the model and db are loaded before handling requests
@app.before_request
def check_backend_ready():
    # Only guard non-health requests
    if request.path != "/health":
        if model is None or site_visits_col is None:
            return "Service initializing, try again later", 503



# Root - Serve HTML and handle API requests
@app.route("/", methods=["GET"])
def home():
    # Log site visit
    visit_data = {
        "timestamp": datetime.datetime.now(),
        "ip": request.remote_addr,
        "user_agent": request.headers.get("User-Agent")
    }
    site_visits_col.insert_one(visit_data)
    print("Site visit logged:", visit_data)

    # Serve index.html
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "web")
    return send_from_directory(html_path, "index.html")


# THIS IS THE MAIN API ENDPOINT
@app.route("/", methods=["POST"])
@limiter.limit("10 per minute")
def predict():

    # Don't crash the server on bad input
    try:

        # Get JSON data
        data = request.get_json(force=True)
        comment = data.get("text", "")

        # Validate input
        if not comment:
            return jsonify({"error": "No text provided"}), 400
        if not isinstance(comment, str):
            return jsonify({"error": "Text must be a string"}), 400
        if len(comment) > 2000:
            return jsonify({"error": "Text too long, max 2000 characters"}), 400

        # Keep original for logging upon success
        keep_original = comment

        # Preprocess
        tokens = word_tokenize(comment)
        tokens = sanitizer(tokens)
        processed = " ".join(tokens)
        # This preprocessing must match what was done during training
        seq = tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(seq, maxlen=10)

        # Predict
        # There will always be a batch of size 1
        prob = float(model.predict(padded)[0][0])
        # 0.5 seems to work well as a threshold
        # We can update this later if needed
        label = "Toxic (-)" if prob >= 0.5 else "Non-toxic (+)"

        # Log API query
        # always use this schema
        query_data = {
            "timestamp": datetime.datetime.now(),
            "ip": request.remote_addr,
            "user_agent": request.headers.get("User-Agent"),
            "comment": keep_original,
            "label": label,
            "confidence": prob
        }
        api_queries_col.insert_one(query_data)
        print("API query logged:", query_data)

        return jsonify({
            "comment": comment,
            "label": label,
            "confidence": prob
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run Server
if __name__ == "__main__":
    # The port number is set by Sliplane automatically when it's deployed.
    # But it should be 8080
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
