from flask import Flask, request, render_template
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the trained model
with open('trained_logreg_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

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
        data = pd.read_csv(file)
        # Add your preprocessing steps here
        # predictions = model.predict(data)
        
        # For now, we simulate some predictions
        predictions = model.predict(data)  # Replace with your actual prediction logic
        data['Predictions'] = predictions
        
        # Generate a plot (example)
        fig, ax = plt.subplots()
        data['Predictions'].value_counts().plot(kind='bar', ax=ax)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('results.html', tables=[data.to_html(classes='data')], plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
