# BART Summarization Model

## Overview

This project utilizes the BART model (`facebook/bart-base`) for text summarization. The model has been trained on a reduced CNN/Daily Mail dataset (70% of the original size) to generate concise summaries from lengthy articles. Due to the large model size (approx. 600MB), it cannot be uploaded to GitHub. Instead, a link to the saved model on Google Drive is provided.

**Dataset Link:** [CNN/Daily Mail Dataset] https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

**Note:** You might need additional Colab units; start with 20% to 30% of the dataset for efficiency.

## Setup

### Required Packages

Install the necessary Python packages using `pip`:

```bash
pip install transformers
pip install torch==2.2.2
pip install pandas
pip install matplotlib
pip install numpy
pip install seaborn
pip install tensorflow
```

Project Structure
datamodel_train_test.py: Contains code for data processing, model training, and testing.
text_summarization/: Django web application folder.
text_summarization/: Contains the Django project files.
models/: Directory with the saved trained BART model.
Saved Model Link: https://drive.google.com/drive/folders/1ImkdAneUkKKMtZzkD99f2H-wQoX5YVsE?usp=sharing


After Training: Place the saved model as follows:

text_summarization/text_summarization/models/saved_model
saved_tokenizer
Training
Model: BART (facebook/bart-base) for text summarization.
Epochs: 5
Optimizer: AdamW with a learning rate of 5e-5.
Loss Monitoring: Training and validation losses are tracked for performance evaluation.
Testing
Model Loading: Load the saved model and tokenizer from the specified directories.
Summary Generation: Use the generate_summary function to process input articles and generate summaries interactively.
Usage
Train the Model: Run the datamodel_train_test.py script to train the BART model on the dataset.

Test the Model: Use the interactive testing function in the same script to generate summaries for new articles.

Django Application: Deploy the model using the Django app for a web-based interface.

Additional Details
Data: Utilizes a reduced CNN/Daily Mail dataset for efficiency.
Interactive Testing: Allows real-time input of articles and generation of summaries.
Feel free to adjust any details according to your specific needs!

LinkedIn: https://linkedin.com/in/muhammad-hasnain-kayani-820599273

Email: muhammadhasnainkayani@gmail.com

