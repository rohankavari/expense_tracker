import torch
from transformers import BertTokenizer, BertForSequenceClassification

def get_sms_type(input_text,input_sender):
    # Load the trained model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    model.load_state_dict(torch.load('models/sms_classifier_model.pth',map_location=torch.device('cpu') ))
    model.eval()

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define max sequence length
    max_length = 128
    # Preprocess the input text and sender
    text_with_sender = input_sender + " " + input_text
    inputs = tokenizer.encode_plus(
        text_with_sender,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    # Move inputs to appropriate device (CPU or GPU)
    inputs = {key: val for key, val in inputs.items()}

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get predicted class
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class
