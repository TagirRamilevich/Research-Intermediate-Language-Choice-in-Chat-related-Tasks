"""
Use this code to train model using BERT.
"""

tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base') 
labels = {'prob_B': 0,
          'prob_NB': 1,
          'prob_PB': 2
          }

class MT5Classifier(nn.Module):
    def __init__(self, num_classes):
        super(MT5Classifier, self).__init__()
        self.model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base', return_dict=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        hidden_state = outputs.encoder_last_hidden_state[:, 0, :]
        hidden_state = self.dropout(hidden_state)
        logits = self.classifier(hidden_state)
        return logits


def preprocess_input(text):
    tokens = tokenizer.encode(text, padding=True, truncation=True, return_tensors='pt')

    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask

    return input_ids, attention_mask

class DatasetMT5(torch.utils.data.Dataset):
    def __init__(self, df, language, tokenizer):
        self.labels = df['label'].map(labels)
        self.labels = [labels[label] for label in df['label']]
        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in df[f'translation_{language}']]
        self.decoder_inputs = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in df[f'translation_{language}']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_texts = self.texts[idx]
        batch_decoder_inputs = self.decoder_inputs[idx]
        batch_y = self.labels[idx]

        return {
            'input_ids': batch_texts['input_ids'].squeeze(0),
            'attention_mask': batch_texts['attention_mask'].squeeze(0),
            'decoder_input_ids': batch_decoder_inputs['input_ids'].squeeze(0)
        }, batch_y

def evaluate(model, test_data, language):
    test = DatasetMT5(test_data, language, tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()

    total_loss = 0
    total_predictions = []
    total_labels = []

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_input = {key: value.to(device) for key, value in test_input.items()}
            test_label = test_label.to(device)
            input_ids = test_input['input_ids'].squeeze(1)
            decoder_input_ids = test_input['decoder_input_ids'].squeeze(1)
            attention_mask = test_input['attention_mask']

            output = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, attention_mask=attention_mask)

            loss = criterion(output, test_label.long())
            total_loss += loss.item()

            predictions = torch.argmax(output, dim=1).cpu().numpy()
            labels = test_label.cpu().numpy()

            total_predictions.extend(predictions)
            total_labels.extend(labels)

    average_loss = total_loss / len(test_dataloader)
    f1_weighted = f1_score(total_labels, total_predictions, average='weighted')

    model.train()  

    return f1_weighted
