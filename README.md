# Loan Scoring MVP

A Minimum Viable Product (MVP) for loan application scoring using machine learning.

## Features

- Loan application scoring using XGBoost model
- REST API for predictions
- Input validation and preprocessing
- Confidence scoring for predictions

## Project Structure

```
loan_scoring_split_files-template2/
├── data/                  # Data directory
├── model/                 # Trained model files
├── src/
│   ├── mvp.py            # Core MVP implementation
│   ├── api.py            # FastAPI implementation
│   └── train_model.py    # Model training script
└── README.md             # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model (if needed):
```bash
python src/train_model.py
```

## Usage

### Running the API

Start the API server:
```bash
python src/api.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

1. **Predict Loan Application**
   - Endpoint: `POST /predict`
   - Input:
     ```json
     {
         "employment_status": "Full-Time",
         "employer_tenure_years": 5.0,
         "monthly_net_income": 15000000,
         "housing_status": "Own",
         "dti_ratio": 0.35,
         "credit_score": 750,
         "delinquencies_30d": 0,
         "industry_unemployment_rate": 0.05,
         "income_gap_ratio": 0.1
     }
     ```
   - Output:
     ```json
     {
         "probability": 0.85,
         "decision": "Approve",
         "confidence": "High"
     }
     ```

2. **Health Check**
   - Endpoint: `GET /health`
   - Output:
     ```json
     {
         "status": "healthy",
         "model_loaded": true
     }
     ```

### Using the MVP Directly

```python
from src.mvp import LoanScoringMVP

# Initialize the MVP
mvp = LoanScoringMVP()

# Load the model
mvp.load_model()

# Example input data
example_data = {
    'employment_status': ['Full-Time'],
    'employer_tenure_years': [5.0],
    'monthly_net_income': [15000000],
    'housing_status': ['Own'],
    'dti_ratio': [0.35],
    'credit_score': [750],
    'delinquencies_30d': [0],
    'industry_unemployment_rate': [0.05],
    'income_gap_ratio': [0.1]
}

# Make prediction
prediction = mvp.predict(pd.DataFrame(example_data))
print(f"Probability of approval: {prediction[0]:.2%}")
```

## Model Details

The MVP uses an XGBoost classifier with the following features:

- Numerical Features:
  - employer_tenure_years
  - monthly_net_income
  - dti_ratio
  - credit_score
  - delinquencies_30d
  - industry_unemployment_rate
  - income_gap_ratio

- Categorical Features:
  - employment_status
  - housing_status

## Future Improvements

1. Add more detailed feature explanations
2. Implement batch prediction endpoint
3. Add model versioning
4. Add input validation rules
5. Implement logging and monitoring
6. Add authentication and rate limiting
7. Add model retraining pipeline
