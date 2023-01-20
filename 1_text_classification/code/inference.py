# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

JSON_CONTENT_TYPE = 'application/json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_fn(model_dir):

    
    model_dir = '/opt/ml/model/'
    dir_contents = os.listdir(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", return_dict=False
    )
    model.load_state_dict(torch.load(model_dir + "/checkpoint.pt"))
    model = model.eval().to(device)
    
    return (model, tokenizer)


def input_fn(serialized_input_data, content_type=JSON_CONTENT_TYPE):
    print("inside input function::",serialized_input_data)
    if content_type == JSON_CONTENT_TYPE:
        input_data = json.loads(serialized_input_data)
        
        return input_data
        
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + content_type)
        return
    

def predict_fn(input_data, models):
    print("inside predict function::",input_data)
    model_bert, tokenizer = models
    sequence_0 = input_data
    
    max_length = 128
    tokenized_sequence_pair = tokenizer.encode_plus(sequence_0,
                                                    max_length=max_length,
                                                    padding='max_length',
                                                    truncation=True,
                                                    return_tensors='pt').to(device)
    
    
    # Convert example inputs to a format that is compatible with TorchScript tracing
    example_inputs = tokenized_sequence_pair['input_ids'], tokenized_sequence_pair['attention_mask']
    
    with torch.no_grad():
        review_classification_logits = model_bert(*example_inputs)
    
    classes = ['negative', 'positive']
    review_prediction = review_classification_logits[0][0].argmax().item()
    out_dict = {}
    out_dict["text"] = sequence_0
    out_dict["prediction"]= classes[review_prediction]
    
    return out_dict


def output_fn(prediction_output, accept=JSON_CONTENT_TYPE):
    if accept == JSON_CONTENT_TYPE:
        return json.dumps(prediction_output), accept
    
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)
    