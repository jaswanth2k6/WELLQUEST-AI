

\# WELLQUEST AI

\### Student Wellbeing Risk Pattern Assessment and Support System



\*\*Tagline:\*\* Your Wellbeing. Your Journey.



\## About the Project



WELLQUEST AI is an AI and Machine Learning-based student wellbeing assessment and support platform.



The system analyzes student-related factors such as anxiety, sleep quality, academic performance, social support, depression and peer pressure to identify patterns associated with stress levels.



It provides students with assessment results, interactive visualizations, personal insights, progress tracking and support resources. A separate Counselor Portal provides assessment reviews, model analytics and reports.



\## Problem Statement



Students may experience academic pressure, anxiety, sleep difficulties, social challenges and other factors that can affect their wellbeing. They may lack an accessible platform to understand these patterns and find appropriate support.



WELLQUEST AI aims to make student wellbeing assessment information easier to understand through machine learning, visualization and accessible support resources.



\## Proposed Solution



WELLQUEST AI combines machine learning-based assessment with an interactive web application.



The platform includes:



\- Student wellbeing assessment.

\- Machine learning-based stress-pattern prediction.

\- Interactive dashboards and charts.

\- Personal insights and feature analysis.

\- Growth Journey with goals and achievements.

\- Support Hub with practical guidance and support resources.

\- Counselor Dashboard.

\- Model Analytics and model comparison.

\- Student Reviews.

\- Report generation.



\## Main Technologies



| Technology | Purpose |

|---|---|

| Python | Core programming language |

| Streamlit | Web application framework |

| Pandas | Data handling |

| NumPy | Numerical operations |

| Scikit-learn | Machine learning |

| Plotly | Interactive visualizations |

| Joblib | Saving and loading trained models |



\## Machine Learning Algorithms



The project uses two classification algorithms:



1\. Logistic Regression

2\. Random Forest Classifier



Both models were trained using student stress-related dataset features.



\### Preliminary Model Performance



| Model | Accuracy |

|---|---:|

| Logistic Regression | 88.18% |

| Random Forest | 88.18% |



These results are from a preliminary train-test evaluation and should not be interpreted as clinical or real-world validation.



\## Project Structure



```text

WELLQUEST\_AI/

│

├── app.py

├── train\_model.py

├── test\_models.py

├── requirements.txt

├── README.md

│

├── data/

│   └── StressLevelDataset.csv

│

├── models/

│   ├── logistic\_regression.joblib

│   └── random\_forest.joblib

│

└── reports/

&#x20;   ├── feature\_importance.csv

&#x20;   ├── Logistic\_Regression\_confusion\_matrix.csv

&#x20;   ├── Random\_Forest\_confusion\_matrix.csv

&#x20;   └── model\_comparison.json

```



\## Installation and Execution



\### 1. Clone the repository



```bash

git clone YOUR\_GITHUB\_REPOSITORY\_URL

```



\### 2. Open the project folder



```bash

cd WELLQUEST\_AI

```



\### 3. Create a virtual environment



```bash

python -m venv .venv

```



\### 4. Activate the environment



Windows:



```bash

.venv\\Scripts\\activate

```



\### 5. Install dependencies



```bash

pip install -r requirements.txt

```



\### 6. Run the application



```bash

streamlit run app.py

```



The application will open in your browser.



\## How to Use



1\. Open the WELLQUEST AI application.

2\. Enter the Student Portal.

3\. Complete the wellbeing assessment.

4\. View the prediction and assessment dashboard.

5\. Explore personal insights and Growth Journey.

6\. Visit the Support Hub for guidance and support resources.

7\. Counselors can use the Counselor Portal to review available assessment information and model analytics.



\## Future Enhancements



\- AI-guided conversational wellbeing assessment.

\- Adaptive follow-up questions.

\- Natural-language assessment explanations.

\- Explainable AI insights.

\- More personalized recommendations.



\## Important Disclaimer



WELLQUEST AI is an educational wellbeing-support prototype. It is not a medical diagnostic system and must not replace professional mental health assessment or counselling.



Model predictions reflect patterns learned from the available dataset and are not guaranteed to represent an individual student's actual mental health condition.



\## Project Status



Phase 1: Core platform development completed.



Phase 2: Advanced AI-guided assessment and enhancements planned.



\---



\*\*WELLQUEST AI – Your Wellbeing. Your Journey.\*\*



