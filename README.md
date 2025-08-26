Semantic Analysis Web App

This project is a containerized web application for semantic analysis.
It includes a frontend, backend API, MongoDB database, tokenizer, and machine learning model.
Everything is orchestrated via Docker Compose.

Accessing the website: https://semanticanalysis.sliplane.app

Querying the api: Send a POST request to the website with the comment wrapped in a JSON structure mapping "text" to your comment as a string.

Project Structure:

* app.py - Main FastAPI/Flask app (serves endpoints)
* script.py - ML model training script
* api_query_example.py - Example of a free api call
* tokenizer.pkl - Pre-fitted tokenizer for text preprocessing
* web Directory - Contains the frontend information
* Data Directory - Contains the training data
* requirements.txt - Contains the requirements necessary for the virtual environment
* Dockerfile - Builds the API / app image
* docker-compose.yml - Defines services: API, frontend, MongoDB for local development

Running the Project LOCALLY:

1. Clone repo & build containers:
   git clone <repo-url>
   cd <repo>
   Setup your virtual environment with Docker
   Download the GloVe word embeddings 6B.300d if you wish to train the model yourself or run the script: https://huggingface.co/sentence-transformers/average_word_embeddings_glove.6B.300d
   Change the `mongo_uri` variable in `app.py` to your database container
   docker compose up --build

2. Access services:
   Frontend: http://localhost:8080
   MongoDB: running on port 27017

3. Training / Updating Model:
   Run script.py to retrain or update the ML model
   python script.py

Notes:
- Make sure your MongoDB service is running before starting the API
- All containers share a Docker network
- The Dockerfile has a direct link to where the ML model is stored; you may download and run it locally if you wish