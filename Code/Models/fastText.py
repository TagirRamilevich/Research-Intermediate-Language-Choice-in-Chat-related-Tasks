"""
Use this code to train model using fastText.
"""

def train_models(language, translation_type):
    """Train a fastText model and print scores."""
    os.chdir(f'/content/drive/My Drive/DBDC_data/translation_files/')

    df_train = pd.read_csv(f'DBDC_dev_translated_{translation_type}_file_{language}.csv')
    df_train[f'translation_{language}'] = df_train[f'translation_{language}'].astype(str)
    df_train[f'translation_{language}'] = df_train[f'translation_{language}'].apply(clean_text)

    df_test = pd.read_csv(f'DBDC_eval_translated_{translation_type}_file_{language}.csv')
    df_test[f'translation_{language}'] = df_test[f'translation_{language}'].astype(str)
    df_test[f'translation_{language}'] = df_test[f'translation_{language}'].apply(clean_text)

    with open(f'train_{translation_type}_{language}.txt', 'w') as f:
        for each_text, each_label in zip(df_train[f'translation_{language}'], df_train['label']):
            f.writelines(f'__label__{each_label} {each_text}\n')
            
    with open(f'test_{translation_type}_{language}.txt', 'w') as f:
        for each_text, each_label in zip(df_test[f'translation_{language}'], df_test['label']):
            f.writelines(f'__label__{each_label} {each_text}\n')

    model = fasttext.train_supervised(f'train_{translation_type}_{language}.txt')
    
    print('Language: ', language, '\n', 'Translation type: ', translation_type, sep='')
    print_results(*model.test(f'test_{translation_type}_{language}.txt'))
    print('\n')

def load_vectors(file_name):
    """Load downloaded fastText vectors."""
    fin = io.open(file_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data

def create_dataset(csv_path, model_path, language):
    model = fasttext.load_model(model_path)
    df = pd.read_csv(csv_path)
    X = np.array([model.get_sentence_vector(sentence) for sentence in df[f'translation_{language}']]) #
    y = np.array(df['label'])
    return X, y
    
def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    vec = fasttext_model.get_sentence_vector(text)
    return vec

class TextDataset(Dataset):
    def __init__(self, df, language):
        self.texts = df[f'translation_{language}'] #
        self.labels = df['label']
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return preprocess(text), label
        
class FastTextClassifier(nn.Module):
    """Define the model architecture."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        x = self.fc1(text)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x       

def evaluate(model, iterator):
    """Evaluate the model."""
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, label = batch
            predictions = model(text)
            loss = criterion(predictions, label)
            acc = (predictions.argmax(1) == label).float().mean()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def accuracy(preds, y):
    """Define accuracy function."""
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    """Train function."""
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, label = batch
        predictions = model(text)
        loss = criterion(predictions, label)

        acc = accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
