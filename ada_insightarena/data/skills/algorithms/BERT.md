


# BERT Implementation Pipeline



Required Imports


``` python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# optimizer from hugging face transformers
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
```



Data Preparation


``` python
def prepare_data(data_path):
    data = pd.read_csv(data_path)
    # Display some basic information
    print("Shape of the data: ", data.shape)
    print(data.head())
    train_text, temp_text, train_labels, temp_labels = train_test_split(data['text'], data['label'],random_state=2018,test_size=0.3, stratify=data['label'])
    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, random_state=2018, test_size=0.5, stratify=temp_labels)
    return train_text, val_text, test_text, train_labels, val_labels, test_labels
```



Data Manipulation- Data Loading


``` python

def transform_data(data,train_text,test_text,val_text,train_labels,test_labels,val_labels,batch_size):
  bert = AutoModel.from_pretrained('bert-base-uncased')
  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
  tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True)
  tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True)
  tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True)
  train_seq = torch.tensor(tokens_train['input_ids'])
  train_mask = torch.tensor(tokens_train['attention_mask'])
  train_y = torch.tensor(train_labels.tolist())
  val_seq = torch.tensor(tokens_val['input_ids'])
  val_mask = torch.tensor(tokens_val['attention_mask'])
  val_y = torch.tensor(val_labels.tolist())
  test_seq = torch.tensor(tokens_test['input_ids'])
  test_mask = torch.tensor(tokens_test['attention_mask'])
  test_y = torch.tensor(test_labels.tolist())

  train_data = TensorDataset(train_seq, train_mask, train_y)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

  val_data = TensorDataset(val_seq, val_mask, val_y)
  val_sampler = SequentialSampler(val_data)
  val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
```


BERT Implementation


``` python
# freeze all the parameters
for param in bert.parameters():
    param.requires_grad = False

class BERT_Arch(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.relu =  nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

        #pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)
        return x
# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)
```

Visualization


``` python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to visualize results
def visualize_results(predictions, labels):
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
```


Evaluate Results


``` python
# Function to evaluate the model
def evaluate_model(model, test_dataloader):
    model.eval()
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            batch = [item.to(device) for item in batch]
            sent_id, mask, labels = batch

            # Get predictions
            preds = model(sent_id, mask)
            preds = torch.argmax(preds, axis=1)

            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())

    # Generate a classification report
    report = classification_report(total_labels, total_preds, target_names=['Class 0', 'Class 1'])
    print("\nClassification Report:\n", report)
    return total_preds, total_labels
```



Save Results


``` python
# Function to save results to a file
def save_results(predictions, labels, output_path):
    results_df = pd.DataFrame({'Actual': labels, 'Predicted': predictions})
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
```


Main Execution


``` python
def main():
    # Path to dataset and output files
    DATA_PATH = 'path_to_your_data.csv'
    OUTPUT_PATH = 'bert_results.csv'
    BATCH_SIZE = 16

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Data Preparation
    print("\nPreparing data...")
    train_text, val_text, test_text, train_labels, val_labels, test_labels = prepare_data(DATA_PATH)

    # Step 2: Data Manipulation
    print("\nTransforming data...")
    transform_data(
        data=DATA_PATH,
        train_text=train_text,
        test_text=test_text,
        val_text=val_text,
        train_labels=train_labels,
        test_labels=test_labels,
        val_labels=val_labels,
        batch_size=BATCH_SIZE
    )

    # Step 3: Define Model Architecture
    print("\nDefining model architecture...")
    bert = AutoModel.from_pretrained('bert-base-uncased')
    model = BERT_Arch(bert).to(device)

    # Step 4: Optimizer and Loss Function
    optimizer = AdamW(model.parameters(), lr=1e-5)
    cross_entropy = nn.NLLLoss()

    # Training Loop
    print("\nStarting training...")
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            batch = [item.to(device) for item in batch]
            sent_id, mask, labels = batch

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

    # Step 5: Evaluation
    print("\nEvaluating the model...")
    predictions, labels = evaluate_model(model, val_dataloader)

    # Step 6: Save Results
    print("\nSaving results...")
    save_results(predictions, labels, OUTPUT_PATH)

    # Step 7: Visualization
    print("\nVisualizing results...")
    visualize_results(predictions, labels)

if __name__ == "__main__":
    main()
```



``` python
```

