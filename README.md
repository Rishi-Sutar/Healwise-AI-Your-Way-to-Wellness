
# Healwise - AI Your Way to Wellness

Healwise-AI is a health diagnostic tool that leverages a Support Vector Classifier (SVC) model trained on Training.csv to predict potential diseases based on user-reported symptoms. After making a prediction, the application provides detailed health advice, including descriptions, diets, medications, and workouts related to the predicted disease.

![App Screenshot](https://github.com/Rishi-Sutar/Healwise-AI-Your-Way-to-Wellness/blob/master/images/Screenshot%20(4).png)


## Features

- Symptom Analysis: Input your symptoms to receive disease predictions.
- Health Recommendations: Get detailed advice including diet, medications, and workouts based on the predicted disease.
- Interactive UI: User-friendly interface built with Streamlit.
- Accurate Predictions: Model achieves an accuracy of 84%.

## Installation

To set up Healwise-AI on your local machine using Conda, follow these steps:

- Clone the repository:

```bash
git clone https://github.com/yourusername/Healwise-AI-Your-Way-to-Wellness.git
cd Healwise-AI-Your-Way-to-Wellness
```

- Create and activate a Conda environment:

```bash
conda create -n healwise-ai python=3.9
conda activate healwise-ai
```

- Install required packages:

```bash
pip install -r requirements.txt
```
    

## Usage

#### Data Ingestion, Transformation, and Model Training

- Ensure you have the Training.csv file in the correct path as specified in main.py. This file is used to train the machine learning model.

- Run the following command to ingest data, transform it, and train the model:

```bash
python main.py
```

This will generate a model.pkl file in the artifacts folder.

#### Running the Streamlit Application

- Launch the Streamlit app:

```bash
streamlit run app.py
```
- Open your browser and navigate to the local address provided by Streamlit (usually http://localhost:8501) to start using the application.


## Dataset

#### CSV Files for Model Building
- Training.csv: Contains the training data used to build the machine learning model. Ensure this file is properly formatted and placed as specified in main.py.

#### CSV Files for Health Recommendations
The following CSV files are used to provide detailed recommendations after disease prediction:

- Symptom-severity.csv: Contains severity ratings of symptoms.
- data.csv: General dataset used for various processes.
- description.csv: Provides descriptions of diseases.
- diets.csv: Contains recommended diets associated with each disease.
- medications.csv: Lists medications for treating different diseases.
- precautions_df.csv: Details precautions to be observed for each disease.
- symtoms_df.csv: Detailed data on symptoms.
- test.csv: Data used for testing the model’s performance.
- train.csv: Data used for training the model.
- workout_df.csv: Provides workout recommendations for each disease.
Ensure these files are correctly formatted and located in the appropriate directory for the Streamlit app to access and display relevant information.
## How it Works

- Data Ingestion: The raw data from Training.csv is loaded to train the model. Other CSV files are used to display detailed information post-prediction.
- Transformation: The data is preprocessed and transformed into a format suitable for model training.
- Training: The SVC model is trained on the processed Training.csv data and saved as model.pkl.
The saved model.pkl file is then loaded by the Streamlit app to make predictions based on user inputs.
## License

This project is licensed under the GPU License. See the LICENSE file for details.

