from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import os
import shutil

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")



df = pd.read_csv("reviews.csv")
df = df.sample(frac=1).reset_index(drop=True)
df.sample()

def convert2num(value):
    if value=='positive': 
        return 1
    else: 
        return 0
    
df['sentiment']  =  df['sentiment'].apply(convert2num)
train = df[:22500]
test = df[22500:].reset_index(drop=True)


def convert2inputexamples(train, test, review, sentiment): 
    trainexamples = train.apply(lambda x:InputExample(guid=None, text_a = x[review], label = x[sentiment]), axis = 1)
    validexamples = test.apply(lambda x: InputExample( guid=None, text_a = x[review],label = x[sentiment]), axis = 1)
  
    return trainexamples, validexamples
trainexamples, validexamples = convert2inputexamples(train,  test, 'review',  'sentiment')

def convertexamples2tf(examples, tokenizer, max_length=128):
    features = []
    for i in tqdm(examples):
        input_dict = tokenizer.encode_plus(
            i.text_a,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True
        )
        input_ids, token_type_ids, attention_mask = (
            input_dict["input_ids"],
            input_dict["token_type_ids"],
            input_dict['attention_mask']
        )
        features.append(InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label=i.label
        ))

    def generate():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        generate,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )

DATA_COLUMN = 'review'
LABEL_COLUMN = 'sentiment'

train_dataset = convertexamples2tf(trainexamples, tokenizer)
valid_dataset = convertexamples2tf(validexamples, tokenizer)

# Assuming you have a list of examples
# Each example should have 'text_a' and 'label' attributes
# Train the model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model
model.fit(train_dataset.shuffle(100).batch(32).repeat(2), epochs=2, validation_data=valid_dataset.batch(32))

# Save the trained model
model.save_pretrained('sentiment_model')
tokenizer.save_pretrained('sentiment_model')


