from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import joblib
import os
import pickle

vectorizer_path = "models/vectorizer.pkl"

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

class InvoiceMatchingSystem:
    def __init__(self, model=None, vectorizer=None):
        self.model = model or PassiveAggressiveClassifier(max_iter=1000, random_state=42)
        self.vectorizer = vectorizer
        self.feedback_buffer = []
    
    def get_next_prediction(self, invoice1, invoice2):
        """Get prediction for a single pair"""
        X = self.vectorizer.transform([f"{invoice1} | {invoice2}"])
        # prob = self.model.predict_proba(X)[0]
        prediction = self.model.predict(X)[0]
        return {
            'Data Set 1': invoice1,
            'Data Set 2': invoice2,
            'predicted_label': prediction,
            # 'confidence': prob[prediction],
            'similarity_score': cosine_similarity(
                self.vectorizer.transform([invoice1]), 
                self.vectorizer.transform([invoice2])
            )[0][0]
        }
    
    def update_model(self, invoice1, invoice2, user_label):
        """Update model with user feedback"""
        X = self.vectorizer.transform([f"{invoice1} | {invoice2}"])
        self.model.partial_fit(X, np.array([user_label]), classes=np.array([0, 1, 2]))
        
        # Save feedback for later analysis
        self.feedback_buffer.append({
            'Data Set 1': invoice1,
            'Data Set 2': invoice2,
            'user_label': user_label,
            'timestamp': pd.Timestamp.now()
        })
        
        # Periodically save model
        if len(self.feedback_buffer) % 10 == 0:  # Save every 10 feedbacks
            self.save_model()
    
    def save_model(self):
        """Save model and feedback data"""
        joblib.dump(self.model, "invoice_match_model.pkl")
        
        # Save feedback history
        pd.DataFrame(self.feedback_buffer).to_csv(
            "feedback_history.csv", 
            mode='a', 
            header=not os.path.exists("feedback_history.csv"),
            index=False
        )