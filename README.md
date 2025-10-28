🚨 Violent Crime Risk Prediction (India)
This project implements an end-to-end Machine Learning pipeline to predict the probability of a reported incident falling under the Violent Crime domain based on situational, geographical, and temporal features.
The prediction model is deployed via a Streamlit web application for real-time risk assessment.
📂 Project Structure
crime-predictor-project/
├── app/
│   └── app.py                  # Streamlit web application deployment script.
├── data/
│   ├── raw/
│   │   └── crime_data.csv      # Original raw data (your uploaded file).
│   └── processed/
│       └── processed_crime_data.csv  # Cleaned, engineered data (used for training).
├── models/
│   └── crime_predictor_model.joblib  # Trained Random Forest Classifier model.
├── notebooks/
│   ├── 01_Data_Cleaning_and_Geocoding.ipynb
│   └── 02_Model_Training_and_Evaluation.ipynb
├── requirements.txt            # Python dependencies.
└── README.md                   # This file.


🚀 How to Run the Application
Prerequisites
Clone the Repository:
git clone [YOUR REPO URL]
cd [YOUR REPO FOLDER NAME]


Install Dependencies:
pip install -r requirements.txt


Step 1: Train the Model
The model must be trained first to create the models/crime_predictor_model.joblib file.
Ensure crime_data.csv is in the data/raw folder.
Run the cells in 01_Data_Cleaning_and_Geocoding.ipynb.
Run the cells in 02_Model_Training_and_Evaluation.ipynb (This saves the model file).
Step 2: Launch the Streamlit App
Once the model file is generated, navigate to the project root and launch the app:
streamlit run app/app.py


The application will open in your web browser, allowing you to interact with the model and generate predictions.
