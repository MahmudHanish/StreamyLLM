import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load the trained model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('sentiment_model')
tokenizer = BertTokenizer.from_pretrained('sentiment_model')

# Load the CSV file
df = pd.read_csv('/song_lyrics.csv')

def predict_sentiment(text, model, tokenizer):
    # Preprocess the input text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_tensors='tf',
        pad_to_max_length=True,
        truncation=True
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]
    
    # Make predictions
    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    logits = outputs[0]
    prediction = tf.nn.softmax(logits, axis=-1)
    
    # Interpret the results
    predicted_label = tf.argmax(prediction, axis=1).numpy()[0]
    confidence = prediction.numpy()[0][predicted_label]
    
    return predicted_label, confidence

# Add a column for sentiment
df['Sentiment'] = df['Lyrics'].apply(lambda x: predict_sentiment(x, model, tokenizer)[0])

# Display sentiment for each song
for index, row in df.iterrows():
    print(f"Song: {row['Title']} by {row['Artist']}, Sentiment: {row['Sentiment']}")

# Save the updated DataFrame to a new CSV file
df.to_csv('/song_lyrics_s', index=False)

# Calculate the average sentiment
average_sentiment = df['Sentiment'].mean()
print(f"Average Sentiment: {average_sentiment}")
