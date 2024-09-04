BART Summarization Model

Overview

This project uses the BART model (facebook/bart-base) to perform text summarization. The model is trained on a reduced CNN/Daily Mail dataset (70% of the original size) to generate concise summaries of long articles.

Dataset Link : https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

Setup

Required Packages: 
Install the necessary Python packages using pip:

pip install transformers

pip install torch==2.2.2

pip install pandas

pip install matplotlib

pip install numpy

pip install seaborn

pip install tensorflow


Project Structure

datamodel_train_test.py: Contains code for data processing, model training, and testing.

text_summarization/: Django web application folder.

text_summarization/: Contains the Django project files.

models/: Directory with the saved trained BART model.

Model Size is almost 600Mb soo it cant be uplaoded to github
Saved Model Link: https://drive.google.com/drive/folders/1ImkdAneUkKKMtZzkD99f2H-wQoX5YVsE?usp=sharing

After training place the saved model as below
text_summarization/text_summarization/models/saved_model and saved_tokenizer




Training

Model: BART (facebook/bart-base) for text summarization.
Epochs: 5
Optimizer: AdamW with a learning rate of 5e-5.
Loss Monitoring: Training and validation losses are tracked for model performance evaluation.

Testing

Model Loading: The saved model and tokenizer are loaded from specified directories.
Summary Generation: The generate_summary function processes input articles and produces summaries interactively.

Usage

Train the Model: Run the datamodel_train_test.py script to train the BART model on the dataset.
Test the Model: Use the interactive testing function in the same script to generate summaries for new articles.
Django Application: The Django app can be used for deploying the model in a web interface.

Additional Details

Data: Reduced CNN/Daily Mail dataset for efficiency.
Interactive Testing: Allows users to input articles and receive generated summaries in real-time.

Feel free to adjust any details according to your specific needs!


