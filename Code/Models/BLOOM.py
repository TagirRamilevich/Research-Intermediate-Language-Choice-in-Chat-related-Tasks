"""
Use this code to train model using BLOOM.
"""

tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom-560m')
labels = {'prob_B': 0, 'prob_NB': 1, 'prob_PB': 2}

class BLOOMClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BLOOMClassifier, self).__init__()
        self.model = BloomModel.from_pretrained('bigscience/bloom-560m', return_dict=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        hidden_state = self.dropout(hidden_state)
        logits = self.classifier(hidden_state)
        return logits

def preprocess_input(text):
    encoded_inputs = tokenizer.encode_plus(
        text,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    return {
        'input_ids': input_ids.squeeze(0),
        'attention_mask': attention_mask.squeeze(0),
    }


class DatasetBLOOM(torch.utils.data.Dataset):
    def __init__(self, df, language, tokenizer):
        self.labels = [labels[label] for label in df['label']]
        self.texts = [
            tokenizer(
                text,
                padding='max_length',
                max_length=512,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
                return_attention_mask=True,
                return_token_type_ids=False,
            ) for text in df[f'translation_{language}']
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_texts = self.texts[idx]
        batch_y = self.labels[idx]

        return {
            'input_ids': batch_texts['input_ids'].squeeze(0),
            'attention_mask': batch_texts['attention_mask'].squeeze(0),
        }, batch_y

        
def collate_fn(batch):
    if isinstance(batch[0], tuple):
        batch_texts = [item[0]['input_ids'] for item in batch]
        batch_y = [item[1] for item in batch]
    else:
        batch_texts = [item['input_ids'] for item in batch]
        batch_y = None

    max_length = max(text.size(0) for text in batch_texts)
    batch_texts = [torch.cat([text, torch.zeros(max_length - text.size(0), dtype=torch.long)]) for text in batch_texts]
    batch_texts = torch.stack(batch_texts)

    if batch_y is not None:
        batch_y = torch.tensor(batch_y)

    return {'input_ids': batch_texts, 'attention_mask': batch_texts != 0}, batch_y


def evaluate(model, test_data, language):
    test = DatasetBLOOM(test_data, language, tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model.eval()

    total_loss = 0
    total_predictions = []
    total_labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_input = {key: value.to(device) for key, value in test_input.items()}
            test_label = test_label.to(device)
            input_ids = test_input['input_ids'].squeeze(1)
            attention_mask = test_input['attention_mask']

            output = model(input_ids=input_ids, attention_mask=attention_mask)

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
