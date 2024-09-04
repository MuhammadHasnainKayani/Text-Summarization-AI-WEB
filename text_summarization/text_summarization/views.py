from django.shortcuts import render, redirect
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import os


base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'models', 'bart_cnn_model')
tokenizer_path = os.path.join(base_dir, 'models', 'bart_cnn_tokenizer')

# Load the model and tokenizer (make sure these paths are correct)
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(tokenizer_path)
if model:
    print("model loaded succesfully")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize_text(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=900, truncation=True)
    summary_ids = model.generate(inputs["input_ids"].to(device),
                                 num_beams=4, 
                                 no_repeat_ngram_size=3, 
                                 early_stopping=True,
                                 max_length=180,  # Set a higher max_length
                                 length_penalty=2.0)  # Optional: control the length penalty
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(summary)
    return summary

    
def home(request):
    if request.method == "POST":
        input_text = request.POST.get('input_text')
        if input_text:
            summary = summarize_text(input_text)
            request.session['summary'] = summary
            request.session['input_text'] = input_text  # Save the input text to the session
            return redirect('home')  # Redirect to the same page
    else:
        summary = request.session.pop('summary', None)  # Retrieve and clear the summary
        input_text = request.session.pop('input_text', None)  # Retrieve and clear the input text
    
    return render(request, 'index.html', {'summary': summary, 'input_text': input_text})