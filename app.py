from flask import Flask, request, render_template
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter
import numpy as np
from nltk.stem import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and the TF-IDF vectorizer
with open('trained_logreg_model.sav', 'rb') as model_file:
    logreg = pickle.load(model_file)

with open('tfidf_vectorizer.sav', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Define stop words and stemmer (extracted from the notebook)
stop_words = {'yourselves', 'between', 'whom', 'itself', 'is', "she's", 'up', 'herself', 'here', 'your', 'each',
             'we', 'he', 'my', "you've", 'having', 'in', 'both', 'for', 'themselves', 'are', 'them', 'other',
             'and', 'an', 'during', 'their', 'can', 'yourself', 'she', 'until', 'so', 'these', 'ours', 'above',
             'what', 'while', 'have', 're', 'more', 'only', "needn't", 'when', 'just', 'that', 'were', "don't",
             'very', 'should', 'any', 'y', 'isn', 'who',  'a', 'they', 'to', 'too', "should've", 'has', 'before',
             'into', 'yours', "it's", 'do', 'against', 'on',  'now', 'her', 've', 'd', 'by', 'am', 'from',
             'about', 'further', "that'll", "you'd", 'you', 'as', 'how', 'been', 'the', 'or', 'doing', 'such',
             'his', 'himself', 'ourselves',  'was', 'through', 'out', 'below', 'own', 'myself', 'theirs',
             'me', 'why', 'once',  'him', 'than', 'be', 'most', "you'll", 'same', 'some', 'with', 'few', 'it',
             'at', 'after', 'its', 'which', 'there','our', 'this', 'hers', 'being', 'did', 'of', 'had', 'under',
             'over','again', 'where', 'those', 'then', "you're", 'i', 'because', 'does', 'all'}  # Add your stop words here
ps = PorterStemmer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        df = pd.read_csv(file)
        df['reviewText'] = df['reviewText'].fillna('')  # Handle missing values

        # Text preprocessing
        df['reviewText'] = df['reviewText'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        df['reviewText'] = df['reviewText'].apply(
            lambda x: ' '.join([ps.stem(word) for word in x.split() if word not in stop_words]))

        # Transform the processed reviews using the fitted TfidfVectorizer
        X_new = tfidf_vectorizer.transform(df['reviewText'])

        # Predict the sentiment using the trained Logistic Regression model
        y_pred_new = logreg.predict(X_new)

        # Count the sentiments
        sentiment_counts = Counter(y_pred_new)

        # Extract main concerns (replace with your logic to identify main concerns)
        main_concerns = ["price", "quality", "customer service"]

        # Initialize result table
        result_table = {"Total Positives": sentiment_counts[2], "Total Negatives": sentiment_counts[0]}

        # Count positives and negatives for each main concern
        for concern in main_concerns:
            concern_count = {"Positive": 0, "Negative": 0}
            for i, review in enumerate(df['reviewText']):
                if concern in review:
                    if y_pred_new[i] == 2:
                        concern_count["Positive"] += 1
                    elif y_pred_new[i] == 0:
                        concern_count["Negative"] += 1
            if concern_count["Positive"] + concern_count["Negative"] >= 50:
                result_table[concern] = concern_count

        # Create result DataFrame for display
        result_df = pd.DataFrame(result_table).T

        # Generate plots
        fig, ax = plt.subplots(2, 1, figsize=(10, 12))

        # Overall sentiment distribution
        ax[0].bar(["Positive", "Negative"], [result_table["Total Positives"], result_table["Total Negatives"]])
        ax[0].set_xlabel("Sentiment")
        ax[0].set_ylabel("Count")
        ax[0].set_title("Overall Sentiment Distribution")

        # Sentiment distribution by concern
        concerns = [concern for concern in result_table.keys() if concern not in ["Total Positives", "Total Negatives"]]
        positive_counts = [result_table[concern]["Positive"] for concern in concerns]
        negative_counts = [result_table[concern]["Negative"] for concern in concerns]
        bar_width = 0.35
        index = np.arange(len(concerns))
        ax[1].bar(index, positive_counts, bar_width, label="Positive")
        ax[1].bar(index + bar_width, negative_counts, bar_width, label="Negative")
        ax[1].set_xlabel("Main Concerns")
        ax[1].set_ylabel("Count")
        ax[1].set_title("Sentiment Distribution by Main Concern")
        ax[1].set_xticks(index + bar_width / 2)
        ax[1].set_xticklabels(concerns, rotation=45)
        ax[1].legend()
        plt.tight_layout()

        # Convert plot to PNG image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('results.html', tables=[result_df.to_html(classes='data')], plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
