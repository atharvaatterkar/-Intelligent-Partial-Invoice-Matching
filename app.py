import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import shutil
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for, send_file
import pickle


name = "Invoice"
app = Flask(__name__)

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


vec = joblib.load('models/vectorizer.joblib')
pretrained_model = joblib.load('models/invoice_match_model.joblib')

class InvoiceMatchingSystem:
    def __init__(self, model=pretrained_model, vectorizer=vec):
        self.model = model or PassiveAggressiveClassifier(max_iter=1000, random_state=42)
        self.vectorizer = vectorizer
        self.feedback_buffer = []
    
    def get_next_prediction(self, invoice1, invoice2):
        """Get predictions for pairs of invoices"""
        if not isinstance(invoice1, list) or not isinstance(invoice2, list):
            raise ValueError("invoice1 and invoice2 must be lists")
        
        predictions = []
        for inv1, inv2 in zip(invoice1, invoice2):
            inv1_str = str(inv1)
            inv2_str = str(inv2)
            
            X = self.vectorizer.transform([f"{inv1_str} | {inv2_str}"])
            prediction = self.model.predict(X)[0]
            
            predictions.append({
                'Data Set 1': inv1,
                'Data Set 2': inv2,
                'predicted_label': prediction,
                'similarity_score': cosine_similarity(
                    self.vectorizer.transform([inv1_str]), 
                    self.vectorizer.transform([inv2_str])
                )[0][0]
            })
        
        # Sort predictions by similarity score in descending order
        predictions.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return predictions
    
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

invoice_matching = InvoiceMatchingSystem()

@app.route("/profile", methods=['GET','POST'])
def profile():
    return render_template('profile.html')

@app.route("/setting", methods=['GET','POST'])
def setting():
    return render_template('setting.html')

@app.route("/", methods=['GET','POST'])
def index():
    return render_template('index1.html')
# Routes
@app.route('/help', methods=['GET', 'POST'])
def help():
    return render_template('help.html')


@app.route('/feed', methods=['GET', 'POST'])
def feed():
    """
    Loads the latest prediction.csv file and displays it in feedback.html.
    """
    latest_dir = get_latest_directory()

    if latest_dir is None:
        return jsonify({"error": "No saved datasets found"}), 404

    try:
        # Define prediction.csv path
        prediction_path = os.path.join(latest_dir, "prediction.csv")

        # Check if prediction.csv exists
        if not os.path.exists(prediction_path):
            return jsonify({"error": "No prediction file found"}), 404

        # Read prediction.csv
        prediction_df = pd.read_csv(prediction_path)
        # prediction_html = prediction_df.to_html(classes="table table-bordered", index=False)  # Convert to HTML table
        print(prediction_df)
        return render_template("feed.html", prediction_df=prediction_df)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download-report', methods=['GET','POST'])
def download_report():
    try:
        latest_dir = get_latest_directory()
        if not latest_dir:
            return jsonify({"error": "No reports found"}), 404

        prediction_path = os.path.join(latest_dir, "prediction.csv")
        if not os.path.exists(prediction_path):
            return jsonify({"error": "Report file not found"}), 404

        return send_file(
            prediction_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name='invoice_matches_report.csv'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/input", methods=['GET','POST'])
def save_csv_files():
    """
    Saves two uploaded CSV files to a timestamped directory.

    Returns:
    JSON response indicating success or failure.
    """
    try:
        # Ensure files are provided in the request
        if 'file1' not in request.files or 'file2' not in request.files:
            return {"error": "Missing files. Please upload file1 and file2."}, 400

        file1 = request.files['file1']
        file2 = request.files['file2']

        # Generate directory name with current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_directory = os.path.join("saved_csv_files", timestamp)
        os.makedirs(save_directory, exist_ok=True)

        # Save files in the timestamped directory
       
        file1_path = os.path.join(save_directory, file1.filename)
        file2_path = os.path.join(save_directory, file2.filename)

        file1.save(file1_path)
        file2.save(file2_path)

         # After saving, process the files (trigger /load logic)
        latest_dir = get_latest_directory()

        if latest_dir is None:
            return jsonify({"error": "No saved datasets found"}), 404
        
        try:
            # Define dataset paths
            dataset1_path = os.path.join(latest_dir, "dataset1.csv")
            dataset2_path = os.path.join(latest_dir, "dataset2.csv")

            # Load datasets
            df1 = pd.read_csv(dataset1_path)
            df2 = pd.read_csv(dataset2_path)

            # Extract invoice numbers
            invoices1 = df1["Data Set 1"].astype(str).tolist()
            invoices2 = df2["Data Set 2"].astype(str).tolist()

            # Store the directory timestamp in a variable
            latest_timestamp = os.path.basename(latest_dir)

            prediction = invoice_matching.get_next_prediction(invoices1, invoices2) 

            # Convert prediction to DataFrame and save as CSV
            prediction_df = pd.DataFrame([prediction])  # Wrap the prediction in a list to create a row
            prediction_filename = os.path.join(latest_dir, "prediction.csv")
            prediction_df.to_csv(prediction_filename, index=False)

            print(f"Prediction saved to: {prediction_filename}")

            # Redirect to load function after saving files
            return redirect(url_for('index'))
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        return {"error": str(e)}, 500
    
def get_latest_directory(base_dir="saved_csv_files"):
    """
    Finds the most recent timestamped directory inside base_dir.
    """
    try:
        # Get all directories in base_dir
        all_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        # Sort directories by timestamp (latest first)
        all_dirs.sort(reverse=True)

        # Return the latest directory if exists, else None
        return os.path.join(base_dir, all_dirs[0]) if all_dirs else None
    except Exception as e:
        print(f"Error finding latest directory: {e}")
        return None 
    
@app.route("/predexcel", methods=['GET','POST'])
def predexcel():
    """
    Loads the latest prediction.csv file and displays it in feedback.html.
    """
    latest_dir = get_latest_directory()

    if latest_dir is None:
        return jsonify({"error": "No saved datasets found"}), 404

    try:
        # Define prediction.csv path
        prediction_path = os.path.join(latest_dir, "prediction.csv")

        # Check if prediction.csv exists
        if not os.path.exists(prediction_path):
            return jsonify({"error": "No prediction file found"}), 404

        # Read prediction.csv
        prediction_df = pd.read_csv(prediction_path)
        prediction_html = prediction_df.to_html(classes="table table-bordered", index=False)  # Convert to HTML table

        return render_template("index1.html", prediction_html=prediction_html)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
  app.run(debug=True)


