"""
Use this code to form base dataset.
"""

def clean_text(text):
    """Clean up text from punctuation marks, numbers and other symbols."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text) 
    text = re.sub('\[|\]', '', text)
    text = re.sub('\(|\)', '', text)
    text = re.sub(r'\w*\d+\w*', '', text)
    text = re.sub('\s{2,}', " ", text)
    return text

def parse_json(data):
    """Process JSON files and put them in a DataFrame."""
    parsed_dict = {}

    for column in data['turns']:
      global df

      NB, PB, B = 0, 0, 0
      parsed_dict['turn_index'] = column['turn-index']
      parsed_dict['speaker'] = column['speaker']
      parsed_dict['annonation_id'] = column['annotation-id']
      parsed_dict['utterance'] = column['utterance']
      
      for annotation in column['annotations']:
        if annotation != []:
          if annotation['breakdown'] == 'O':
              NB += 1
          elif annotation['breakdown'] == 'T':
              PB += 1
          elif annotation['breakdown'] == 'X':
              B += 1
              
      parsed_dict['NB'] = NB
      parsed_dict['PB'] = PB
      parsed_dict['B'] = B
      
      if (NB + PB +B) > 0:
        parsed_dict['prob_NB'] = NB * 1.0 / (NB + PB + B)
        parsed_dict['prob_PB'] = PB * 1.0 / (NB + PB + B)
        parsed_dict['prob_B'] = B * 1.0 / (NB + PB + B)
      else:
        parsed_dict['prob_NB'], parsed_dict['prob_PB'], parsed_dict['prob_B'] = 10, 10, 10

      df = df.append(parsed_dict, ignore_index=True, sort=False)

def base_english_dataset(sample):
    """Get development and evaluation datasets."""
    global df
    os.chdir(f'/content/drive/My Drive/DBDC_data/{sample} raw data/')    
    df = pd.DataFrame(columns=['turn_index', 'speaker', 'annonation_id', 'utterance', 'NB', 'PB', 'B', 'prob_NB', 'prob_PB', 'prob_B'])
    df_to_split = pd.DataFrame(columns=['utterance', 'label', 'utterance_A', 'utterance_B'])

    for file in os.listdir(os.curdir):
      if file.endswith('.log.json'):
        with open(file, 'r') as file_to_parse:
          data = json.load(file_to_parse)
          parse_json(data)

    df['utterance'] = df['utterance'].apply(clean_text)
    df['prob_NB'] = pd.to_numeric(df['prob_NB'])
    df['prob_PB'] = pd.to_numeric(df['prob_PB'])
    df['prob_B'] = pd.to_numeric(df['prob_B'])
    df['label'] = df[['prob_B', 'prob_PB', 'prob_NB']].idxmax(axis=1)

    for i in range(len(df.index)-1):    
      parsed_dict = {}
      
      parsed_dict['utterance_A'] = df.iloc[i]['utterance']
      parsed_dict['utterance_B'] = df.iloc[i+1]['utterance']
      parsed_dict['utterance'] = df.iloc[i]['utterance'] + ' ' + df.iloc[i+1]['utterance']
      parsed_dict['label'] = df.iloc[i+1]['label']

      if parsed_dict['utterance_A'] in df_to_split['utterance_B'].values:
        continue
      else:
        df_to_split = df_to_split.append(parsed_dict, ignore_index=True, sort=False)

    df_to_split = df_to_split.drop(['utterance_A', 'utterance_B'], axis=1)
    df_to_split.rename({'utterance': 'translation_en'}, axis=1, inplace=True)
    df_to_split = df_to_split.drop_duplicates(keep='last')
    df_to_split.to_csv(f'/content/drive/My Drive/DBDC_data/translation_files/DBDC_{sample}_translated_original_file_en.csv', index=False)
