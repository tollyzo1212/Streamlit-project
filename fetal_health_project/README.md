# Fetal Health Prediction - Final Project

Files
- Fetal_Health_Prediction.ipynb : full analysis (objectives, cleaning, EDA, scaling, SMOTE, feature selection, models, evaluation)
- app.py + fetal_health_model.joblib : Streamlit web app (keep both in the same folder as fetalhealth.csv is NOT needed for the app)
- Fetal_Health_Presentation.pptx : slides
- fetalhealth.csv : dataset (needed to re-run the notebook)

Run
    pip install -r requirements.txt
    jupyter lab                      # open and run the notebook (top to bottom)
    streamlit run app.py             # launch the web app
