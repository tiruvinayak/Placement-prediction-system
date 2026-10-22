# 🎓 Placement Prediction System

An interactive Machine Learning web application that analyzes student academic and skill-related data to predict placement outcomes. The system provides dataset management, exploratory data analysis, dimensionality reduction, anomaly detection, clustering, and student placement prediction through a web interface.

## 📌 Project Overview

The Placement Prediction System helps analyze student information and estimate placement outcomes using Machine Learning models.

The application provides a centralized interface for uploading datasets, exploring student records, analyzing feature relationships, visualizing patterns, and generating placement predictions using trained models.

## ✨ Features

### 1. Dataset Management
- Upload supported datasets.
- View student records in a tabular format.
- Display actual dataset columns and values.
- Calculate total records and features.
- Identify missing values and duplicate records.
- Generate descriptive statistical summaries.
- Support dynamic dataset inspection.

### 2. Data Visualization
The application includes visualizations for exploring student placement data:

- **Correlation Matrix Heatmap:** Analyzes relationships between numeric features.
- **2D PCA Projection:** Projects high-dimensional numeric data into two dimensions.
- **PCA Scree Plot:** Displays explained variance across principal components.
- **t-SNE Visualization:** Explores nonlinear patterns in student data.
- **Isolation Forest:** Identifies potentially anomalous records.
- **DBSCAN Clustering:** Explores density-based groups within the dataset.

Visualizations depend on the available dataset, valid input features, and successful execution of the corresponding analysis.

### 3. Student Placement Prediction
The prediction interface collects student attributes, including:

- CGPA
- History of Backlogs
- Internships Completed
- Aptitude Test Score
- Academic Projects

The backend processes valid inputs and generates predictions using the configured, trained Machine Learning model.

**Note:** The actual model features, preprocessing requirements, target variable, and prediction output depend on the project's training pipeline and saved model artifacts.

### 4. Machine Learning
The project interface references the following classification algorithms:

- Random Forest Classifier
- Decision Tree Classifier

The application must use the model and preprocessing pipeline that match the actual training data and feature schema.

### 5. Web Interface
- Interactive browser-based interface.
- Dataset upload and summary pages.
- Data visualization dashboard.
- Student prediction form.
- Prediction result panel.
- Responsive, card-based layout.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Flask | Web application backend |
| Pandas | Dataset loading and processing |
| NumPy | Numerical computations |
| Scikit-learn | Machine Learning and data analysis |
| Matplotlib | Statistical visualizations |
| Seaborn | Statistical charts |
| HTML5 | Web page structure |
| CSS3 | Styling and responsive layouts |
| JavaScript | Frontend interactions |

The exact dependencies should match the packages used in the existing project.

## 📂 Project Structure

The following is an illustrative structure. Use the actual filenames and directories present in your repository.

```text
Placement-Prediction-System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   ├── dataset.html
│   ├── visualizations.html
│   └── prediction.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── images/
│
├── data/
│   └── dataset.csv
│
├── models/
│   └── trained_model.pkl
│
└── tests/
    └── test_app.py
```

**Important:** These filenames are examples, not a claim about your actual repository. Keep your existing project structure and update this section to match the files that actually exist.

## ⚙️ Installation and Setup

### Prerequisites

Install the following:

- Python 3.10 or a version supported by the project's dependencies.
- pip
- Git (optional, for cloning the repository)

### 1. Clone the Repository

Replace the example URL with your actual repository URL.

```bash
git clone YOUR_REPOSITORY_URL
cd YOUR_PROJECT_FOLDER
```

If the project is already on your computer, open its existing folder instead.

### 2. Create a Virtual Environment

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

If `requirements.txt` already exists:

```bash
python -m pip install -r requirements.txt
```

Use the existing dependency file rather than installing unnecessary packages or changing compatible versions.

### 4. Configure the Dataset and Model

Ensure that:

- The dataset is available in the location expected by the application, or upload it through the web interface.
- The trained model exists at its configured path.
- Any required encoders, scalers, imputers, and preprocessing artifacts are available.
- The model's expected feature names and order match the prediction pipeline.

Do not rename or replace model artifacts without updating the corresponding configuration and verifying compatibility.

### 5. Run the Application

If the Flask entry point is `app.py`, run:

```bash
python app.py
```

Use the actual entry-point filename if it differs.

Open the local address printed in the terminal. For example:

```text
http://127.0.0.1:5000/
```

The port may differ depending on the existing Flask configuration.

## 🚀 How to Use

1. Start the Flask application.
2. Open the application in your browser.
3. Navigate to Dataset Operations.
4. Upload your student dataset.
5. View the dataset and verify the actual column names and records.
6. Review dataset statistics, including missing values and duplicate records.
7. Open Data Visualizations to explore feature relationships, PCA, t-SNE, anomaly detection, and clustering.
8. Navigate to Student Placement Prediction.
9. Enter the required student attributes.
10. Submit the form and review the prediction returned by the trained model.

## 📊 Dataset Requirements

The uploaded dataset should contain the student attributes required by the application's analysis or training pipeline.

Depending on the model, these may include:

- Academic performance.
- Backlog history.
- Internship experience.
- Aptitude scores.
- Academic project experience.
- A placement-related target column for supervised training.

The exact columns must be determined from the actual dataset and training code.

The application should validate missing values, data types, duplicate records, and feature compatibility before performing analysis or prediction.

## 🧪 Testing and Verification

Verify the following before using the application:

- Dataset uploads complete successfully.
- The dataset table displays the correct columns and values.
- Dataset statistics match the uploaded file.
- Visualizations use the active dataset.
- The prediction pipeline uses the correct features and preprocessing.
- The backend returns valid prediction responses.
- Invalid inputs produce understandable error messages.
- Existing automated tests pass.

Run the project's existing test suite when available. For example:

```bash
python -m pytest
```

This command requires pytest and an appropriately configured test suite.

## 🔍 Troubleshooting

### Dataset is not displayed
- Verify the uploaded file format.
- Check the upload route and dataset-loading logic.
- Inspect the Flask terminal for parsing errors.
- Confirm the template receives the actual dataset records and column names.

### Dataset statistics show placeholders
- Verify that the summary endpoint calculates statistics from the active dataset.
- Check that the frontend renders the returned values.
- Ensure the application does not use stale cached data.

### Visualizations fail
- Check numeric feature selection and missing-value handling.
- Verify that the necessary visualization dependencies are installed.
- Inspect the backend logs for calculation or rendering errors.

### Predictions are incorrect or fail
- Verify the target column and model version.
- Check the expected feature names and order.
- Ensure preprocessing matches the training pipeline.
- Confirm that the correct saved model is loaded.
- Evaluate the model using appropriate held-out data.

### Application does not start
- Activate the correct virtual environment.
- Install the required dependencies.
- Check the entry-point filename.
- Verify the configured port and inspect the terminal error.

## 🔒 Responsible Use

- Validate uploaded files before processing.
- Do not expose sensitive student information publicly.
- Store model artifacts and configuration securely.
- Do not interpret a model prediction as a guaranteed placement outcome.
- Report evaluation metrics honestly.
- Do not display predicted probabilities unless they are supported by the model.

## 🔮 Future Enhancements

Potential improvements include:

- Advanced dataset filtering and search.
- Downloadable dataset summaries.
- Interactive visualization controls.
- Model performance comparison.
- Feature importance analysis.
- Explainable Machine Learning.
- Improved validation and automated testing.
- Deployment to a production server with appropriate security.

## 👨‍💻 Author

**Tiru Vinayak**

B.Tech Student | Machine Learning | Python | Web Development

## 📄 License

Choose an appropriate license for the project before distributing or publishing the repository.

---

**Placement Prediction System** — Exploring student data and estimating placement outcomes through Machine Learning.
