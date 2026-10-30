# ⚡ Indian Power Generation – Machine Learning Analysis

## 📌 About the Project

This project focuses on analyzing Indian power generation data using Machine Learning algorithms. It explores electricity generation patterns, prediction, classification, and clustering using Python and Scikit-learn.

The project uses a dataset containing approximately **32,659 records and 17 columns**, including information about power plants, installed capacity, fuel types, regions, and actual electricity generation.

## 🎯 Project Objectives

- Analyze electricity generation patterns.
- Predict actual electricity generation.
- Classify power generation levels.
- Study power plant categories and sectors.
- Discover hidden patterns using clustering.

## 🤖 Machine Learning Algorithms

Five Machine Learning algorithms are implemented:

| Algorithm | Application |
|---|---|
| Linear Regression | Predict actual electricity generation |
| Logistic Regression | Classify generation into High and Low |
| Decision Tree | Classify power plant categories |
| Support Vector Machine (SVM) | Classify power plant categories |
| K-Means Clustering | Group similar power generation records |

## 🛠️ Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Spyder IDE

## 📂 Project Structure

```text
Indian-Power-Generation-ML-Analysis/
│
├── govdataset.csv
├── power_generation_five_models_pastel.py
└── README.md
```

## 📊 Dataset Information

The dataset contains information related to:

- Power plant categories
- Types of fuel used
- Installed capacity
- Actual energy generated
- Generation programme prepared by CEA
- States and regions
- Power plant sectors
- Year and month information

## ⚙️ Installation

Install the required Python libraries:

```bash
pip install pandas matplotlib scikit-learn
```

## ▶️ How to Run

1. Clone this repository:

```bash
git clone https://github.com/YOUR-USERNAME/Indian-Power-Generation-ML-Analysis.git
```

2. Open the project in Spyder or another Python IDE.

3. Make sure `govdataset.csv` is available in the project folder.

4. Update the dataset file path in the Python script:

```python
file_path = r"X:\OneDrive\Desktop\duck you\govdataset.csv"
```

Replace it with the location of your dataset.

5. Run the Python file.

## 📈 Model Evaluation

The models use appropriate evaluation methods:

- **Linear Regression:** MSE, MAE and R² Score.
- **Logistic Regression:** Accuracy, Precision, Recall and F1 Score.
- **Decision Tree:** Accuracy and Confusion Matrix.
- **SVM:** Accuracy and Confusion Matrix.
- **K-Means:** Cluster distribution and cluster averages.

## 📉 Visualizations

The project includes visual representations of:

- Actual vs Predicted Electricity Generation
- Logistic Regression Confusion Matrix
- Decision Tree Visualization
- SVM Confusion Matrix
- K-Means Clustering

## 🎓 Learning Outcomes

Through this project, I explored:

- Supervised and Unsupervised Learning.
- Regression and Classification.
- Data preprocessing and feature selection.
- Model training and prediction.
- Model evaluation and data visualization.

## 🙏 Acknowledgment

Special thanks to **Ms. Mannet Kaur** for her guidance, support, and teaching throughout this project.

## 👨‍💻 Author

**Rishabh Kumar Yadav**

B.Tech – Computer Science and Engineering

---

⭐ If you find this project useful, feel free to explore and star the repository!
