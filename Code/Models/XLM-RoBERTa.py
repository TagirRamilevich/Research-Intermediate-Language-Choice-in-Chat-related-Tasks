"""
Use this code to train model using XLM-RoBERTa.
"""

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModel.from_pretrained('xlm-roberta-base')

labels = {'prob_B': 0, 'prob_NB': 1, 'prob_PB': 2}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, language):
        self.labels = [labels[label] for label in df['label']]
        self.texts = [
            tokenizer(
                text,
                padding='max_length',
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            for text in df[f'translation_{language}']
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids.squeeze(1),
            attention_mask=attention_mask.squeeze(1),
            return_dict=True
        )
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return logits

def evaluate(model, test_data, language):
    test = Dataset(test_data, language)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_ids = test_input['input_ids'].to(device)
            output = model(input_ids, mask)
            predicted = output.argmax(dim=1)
            true_labels.extend(test_label.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')
    return f1_weighted
