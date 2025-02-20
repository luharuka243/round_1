# Complaint Classification System

A FastAPI-based service for classifying complaints using topic modeling and handling duplicate detection.

## Overview

This system provides a REST API service that:
- Classifies complaints into topics using transformer model developed during the hackathon
- Do the process of the given complaint to the model
- Provides health check and prediction endpoints

## Project Structure

```
.
├── main.py                 # FastAPI application entry point
├── src/
│   ├── model.py           # Classification model implementation
│   └── validator.py       # Input/Output validation schemas
|── model/                 # Contains all the model files
├── supported_script/
│   ├── duplicates_removal.ipynb    # Notebook for removing duplicate complaints
│   └── lda_clustering.ipynb        # Notebook for LDA topic modeling
|   └── text_augumentation.ipynb    # Notebook for text augmentation
|   └── labeller.ipynb              # Notebook for labelling the data
|   └── training.ipynb              # Notebook for training the model
├── test/
│   ├── test.py                     # Test file for bulk testing
|   └── test.json                   # Sample file for the testing
```

## Features

- **REST API Endpoints:**
  - `GET /`: Root endpoint with service description
  - `GET /health`: Health check endpoint
  - `POST /predict`: Main prediction endpoint for complaint classification

- **Duplicate Detection:**
  - Uses MinHash LSH algorithm for efficient similarity detection
  - Configurable similarity threshold
  - Handles both exact and near-duplicate complaints

- **Topic Modeling:**
  - LDA-based topic clustering
  - Automatic validation based on cluster sizes
  - Support for preprocessing and text cleaning

- **Text Augmentation:**
  - Generate more data for the training of the model
  - Use the data to train the model

- **Labeller:**
  - Semi-Supervised learning model to label the data
  - Used the data to train the model

- **Training:**
  - Train the model using the data
  - Save the model for the future use

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the API Server

```bash
python main.py
```

The server will start on `http://0.0.0.0:8000`

### Making Predictions

Send a POST request to `/predict` endpoint:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"complaint_id": "123", "content": "your complaint text here"}'
```

### Deploy within the Docker:

Run Command `docker build -t complaint_classification .`

This will build the docker images in the local.

Run Command `docker run -d -p 8000:8000 --name complaint_classification complaint_classification`

This will start the docker services. where using the same below command will give you result.
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"complaint_id": "123", "content": "your complaint text here"}'
```

### Bulk testing:
Place the file inside the test folder in the same format as given in the test file, where complaint_id is the id of the complaint(Optional) and content is the complaint text.
Run the command `python -m test.test`

### Running Support Scripts

1. For duplicate removal:
   - Open `supported_script/duplicates_removal.ipynb`
   - Update the `FILE_PATH` and `COMPLAIN_COLUMN` variables
   - Run all cells

2. For LDA clustering:
   - Open `supported_script/lda_clustering.ipynb`
   - Follow the notebook instructions
   - Results will be saved in `validated_df.csv` and `non_validated_df.csv`

3. For text augmentation:
   - Open `supported_script/text_augumentation.ipynb`
   - Run all with chnage in the dataframe and category name for which you want to generate more data.

## Configuration

- MinHash LSH parameters:
  - `num_perm`: Number of permutations (default: 100)
  - `num_bands`: Number of bands (default: 20)
  - `threshold`: Similarity threshold for duplicate detection

- LDA parameters:
  - Configurable through the `lda_config` dictionary in the clustering script
  - Adjustable number of topics and other model parameters

## Dependencies

- FastAPI
- Uvicorn
- Loguru
- NLTK
- Gensim
- pandas
- NumPy
- pyLDAvis
- scikit-learn
-  and more

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

None

## Developer

- Lasya Ippangunta
- Shubham Luharuka
- Puneet Hedge

