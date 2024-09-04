################################## Code for reducing dataset to certain threshold line 3 to 39 #####################################

import os
import pandas as pd
import shutil

# Specify the path to your dataset folder
dataset_folder = r'C:\Users\muham\Downloads\Compressed\archive_7\cnn_dailymail'

# Specify the reduction percentage (e.g., 50% or 60%)
# if reduction percentage is 70% then 70% data left and 30% deducted
reduction_percentage = 30  # deducting 70% data

# Create a new folder to save the reduced datasets
reduced_dataset_folder = f'./cnn_dailymail_reduced_{reduction_percentage}percent'
os.makedirs(reduced_dataset_folder, exist_ok=True)


# Function to reduce the dataset
def reduce_dataset(file_name, reduction_percentage):
    file_path = os.path.join(dataset_folder, file_name)
    df = pd.read_csv(file_path)

    # Calculate the number of rows to keep
    reduced_size = int(len(df) * (reduction_percentage / 100))

    # Randomly sample the data
    df_reduced = df.sample(n=reduced_size, random_state=42)

    # Save the reduced dataset to the new folder
    reduced_file_path = os.path.join(reduced_dataset_folder, file_name)
    df_reduced.to_csv(reduced_file_path, index=False)
    print(f"Reduced {file_name} from {len(df)} to {reduced_size} rows.")


# Reduce the train, test, and validation datasets
for file_name in ['train.csv', 'test.csv', 'validation.csv']:
    reduce_dataset(file_name, reduction_percentage)

print(f"Reduced datasets saved in {reduced_dataset_folder}.")









############################################### Code for training model line 50 to 216 With no validation (Takes less time to train) #########################################################

import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

# Paths to your dataset files
train_path = '/content/cnn_dailymail_reduced_70percent/test.csv'
val_path = '/content/cnn_dailymail_reduced_70percent/test.csv'
test_path = '/content/cnn_dailymail_reduced_70percent/test.csv'

# Text cleaning functions
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

def clean_text_vectorized(text_series):
    return text_series.apply(clean_text)

# Function to load and preprocess the data
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    print(f"Initial shape of dataset: {df.shape}")
    print(f"Checking for null values:\n{df.isnull().sum()}\n")

    df = df.drop('id', axis=1)
    df = df[['article', 'highlights']]
    df['article'] = clean_text_vectorized(df['article'])
    df['highlights'] = clean_text_vectorized(df['highlights'])

    # Displaying distribution of text lengths
    df['article_len'] = df['article'].apply(lambda x: len(x.split()))
    df['highlights_len'] = df['highlights'].apply(lambda x: len(x.split()))

    plt.figure(figsize=(10, 5))
    sns.histplot(df['article_len'], bins=50, kde=True, color='blue', label='Article Lengths')
    sns.histplot(df['highlights_len'], bins=50, kde=True, color='orange', label='Highlights Lengths')
    plt.legend()
    plt.title('Distribution of Article and Highlights Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()

    return df

# Load and preprocess datasets
print("Preprocessing Train Dataset")
train_df = load_and_preprocess(train_path)

print("Preprocessing Validation Dataset")
val_df = load_and_preprocess(val_path)

print("Preprocessing Test Dataset")
test_df = load_and_preprocess(test_path)

# Tokenize and Pad Sequences
max_len_text = 900
max_len_summary = 180

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(train_df['article']) + list(train_df['highlights']))

# Text and Summary to sequences for training
train_text_sequences = tokenizer.texts_to_sequences(train_df['article'])
train_summary_sequences = tokenizer.texts_to_sequences(train_df['highlights'])

# Validation sequences
val_text_sequences = tokenizer.texts_to_sequences(val_df['article'])
val_summary_sequences = tokenizer.texts_to_sequences(val_df['highlights'])

# Padding Sequences
train_text_padded = pad_sequences(train_text_sequences, maxlen=max_len_text, padding='post', truncating='post')
train_summary_padded = pad_sequences(train_summary_sequences, maxlen=max_len_summary, padding='post', truncating='post')

val_text_padded = pad_sequences(val_text_sequences, maxlen=max_len_text, padding='post', truncating='post')
val_summary_padded = pad_sequences(val_summary_sequences, maxlen=max_len_summary, padding='post', truncating='post')

# Test sequences (no need to train or validate on these, but keep them ready for inference)
test_text_sequences = tokenizer.texts_to_sequences(test_df['article'])
test_summary_sequences = tokenizer.texts_to_sequences(test_df['highlights'])

test_text_padded = pad_sequences(test_text_sequences, maxlen=max_len_text, padding='post', truncating='post')
test_summary_padded = pad_sequences(test_summary_sequences, maxlen=max_len_summary, padding='post', truncating='post')

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import torch

# Load BART tokenizer and model (lighter version)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Define a custom dataset class
class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries, tokenizer, max_len_text=900, max_len_summary=180):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_len_text = max_len_text
        self.max_len_summary = max_len_summary

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, index):
        article = self.articles[index]
        summary = self.summaries[index]

        inputs = self.tokenizer(article, max_length=self.max_len_text, truncation=True, padding="max_length", return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(summary, max_length=self.max_len_summary, truncation=True, padding="max_length", return_tensors="pt")

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = targets["input_ids"].squeeze()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Create dataset objects
train_dataset = SummarizationDataset(train_df['article'].tolist(), train_df['highlights'].tolist(), tokenizer)
val_dataset = SummarizationDataset(val_df['article'].tolist(), val_df['highlights'].tolist(), tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Set up training configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop (simplified)
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(10):  # You can increase the number of epochs for better performance
    model.train()
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the model
model.save_pretrained('/content/drive/MyDrive/bart_cnn_model')
tokenizer.save_pretrained('/content/drive/MyDrive/bart_cnn_tokenizer')

print("Model saved successfully!")


############################################### Code for training model line 222 to 424 With with validation and testing (Takes large time to train) #########################################################
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

# Paths to your dataset files
train_path = '/content/cnn_dailymail_reduced_30percent/train.csv'
val_path = '/content/cnn_dailymail_reduced_30percent/validation.csv'
test_path = '/content/cnn_dailymail_reduced_30percent/test.csv'

# Text cleaning functions
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

def clean_text_vectorized(text_series):
    return text_series.apply(clean_text)

# Function to load and preprocess the data
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    print(f"Initial shape of dataset: {df.shape}")
    print(f"Checking for null values:\n{df.isnull().sum()}\n")

    df = df.drop('id', axis=1)
    df = df[['article', 'highlights']]
    df['article'] = clean_text_vectorized(df['article'])
    df['highlights'] = clean_text_vectorized(df['highlights'])

    # Displaying distribution of text lengths
    df['article_len'] = df['article'].apply(lambda x: len(x.split()))
    df['highlights_len'] = df['highlights'].apply(lambda x: len(x.split()))

    plt.figure(figsize=(10, 5))
    sns.histplot(df['article_len'], bins=50, kde=True, color='blue', label='Article Lengths')
    sns.histplot(df['highlights_len'], bins=50, kde=True, color='orange', label='Highlights Lengths')
    plt.legend()
    plt.title('Distribution of Article and Highlights Lengths')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.show()

    return df

# Load and preprocess datasets
print("Preprocessing Train Dataset")
train_df = load_and_preprocess(train_path)

print("Preprocessing Validation Dataset")
val_df = load_and_preprocess(val_path)

print("Preprocessing Test Dataset")
test_df = load_and_preprocess(test_path)

# Tokenize and Pad Sequences
max_len_text = 900
max_len_summary = 180

tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(train_df['article']) + list(train_df['highlights']))

# Text and Summary to sequences for training
train_text_sequences = tokenizer.texts_to_sequences(train_df['article'])
train_summary_sequences = tokenizer.texts_to_sequences(train_df['highlights'])

# Validation sequences
val_text_sequences = tokenizer.texts_to_sequences(val_df['article'])
val_summary_sequences = tokenizer.texts_to_sequences(val_df['highlights'])

# Padding Sequences
train_text_padded = pad_sequences(train_text_sequences, maxlen=max_len_text, padding='post', truncating='post')
train_summary_padded = pad_sequences(train_summary_sequences, maxlen=max_len_summary, padding='post', truncating='post')

val_text_padded = pad_sequences(val_text_sequences, maxlen=max_len_text, padding='post', truncating='post')
val_summary_padded = pad_sequences(val_summary_sequences, maxlen=max_len_summary, padding='post', truncating='post')

# Test sequences
test_text_sequences = tokenizer.texts_to_sequences(test_df['article'])
test_summary_sequences = tokenizer.texts_to_sequences(test_df['highlights'])

test_text_padded = pad_sequences(test_text_sequences, maxlen=max_len_text, padding='post', truncating='post')
test_summary_padded = pad_sequences(test_summary_sequences, maxlen=max_len_summary, padding='post', truncating='post')

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(f"Vocabulary size: {vocab_size}")

from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import torch

# Load BART tokenizer and model (lighter version)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# Define a custom dataset class
class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries, tokenizer, max_len_text=900, max_len_summary=180):
        self.articles = articles
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_len_text = max_len_text
        self.max_len_summary = max_len_summary

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, index):
        article = self.articles[index]
        summary = self.summaries[index]

        inputs = self.tokenizer(article, max_length=self.max_len_text, truncation=True, padding="max_length", return_tensors="pt")
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(summary, max_length=self.max_len_summary, truncation=True, padding="max_length", return_tensors="pt")

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = targets["input_ids"].squeeze()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Create dataset objects
train_dataset = SummarizationDataset(train_df['article'].tolist(), train_df['highlights'].tolist(), tokenizer)
val_dataset = SummarizationDataset(val_df['article'].tolist(), val_df['highlights'].tolist(), tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Set up training configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop with validation
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(6):  # You can increase the number of epochs for better performance
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}")

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")

# Save the model
model.save_pretrained('/content/drive/MyDrive/bart_cnn_model')
tokenizer.save_pretrained('/content/drive/MyDrive/bart_cnn_tokenizer')

print("Model saved successfully!")

# Testing
def generate_summary(text):
    inputs = tokenizer(text, max_length=max_len_text, return_tensors="pt", truncation=True, padding="max_length")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    summary_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_len_summary, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example test
test_article = test_df['article'].iloc[0]
print(f"Article: {test_article}")
print(f"Generated Summary: {generate_summary(test_article)}")




################################################# Code for testing model loading saved model and using it  ###############################

from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load the saved model and tokenizer
model_path = '/content/drive/MyDrive/bart_cnn_model'
tokenizer_path = '/content/drive/MyDrive/bart_cnn_tokenizer'

tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_summary(article, max_len_text=900, max_len_summary=180):
    inputs = tokenizer(article, max_length=max_len_text, truncation=True, padding="max_length", return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Generate the summary
    summary_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=max_len_summary, num_beams=4, length_penalty=2.0, early_stopping=True)
    
    # Decode the summary and return it
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def interactive_summary():
    print("Enter your article (or type 'exit' to quit):")
    while True:
        article = input()
        if article.lower() == 'exit':
            break
        summary = generate_summary(article)
        print("\nGenerated Summary:")
        print(summary)
        print("\nEnter another article (or type 'exit' to quit):")

# Run the interactive summary function
interactive_summary()
