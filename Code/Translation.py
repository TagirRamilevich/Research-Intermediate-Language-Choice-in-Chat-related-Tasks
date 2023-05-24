"""
Translation methods: Google Translate and EasyNMT.
"""

def get_official_google_translate(
    text: str,
    source_lang: Enum,
    target_lang: Enum
) -> str:
    """Get translation using Google Translate."""
    home_path = Path().resolve()
    translations_folder = str(Path(home_path, "translation_files"))
    if not os.path.exists(translations_folder):
      os.makedirs(translations_folder, exist_ok=True)
    file_translation = str(Path(translations_folder, f"{uuid.uuid4().hex}.js"))
    text = text.replace('\n', ' ').replace('\'', '\"')

    template = f"""const translate = require('@iamtraction/google-translate');
    translate(
        '{text}',
        {{from: '{source_lang}', to: '{target_lang}' }}).then(res => {{
    console.log(res.text); }}).catch(err => {{
    console.error(err);
    }});
    """
    with open(file_translation, "w", encoding="utf-8") as f:
        f.write(template)
    response = muterun_js(file_translation)

    os.remove(file_translation)
    return response.stdout.decode("utf-8")[:-1] 

def EasyNMT_translation(language, sample):
    """Get translation using EasyNMT."""

    os.chdir('/content/drive/My Drive/DBDC_data/translation_files/')

    model = EasyNMT('opus-mt')
    df_translated = pd.DataFrame(columns=['utterance', 'label', f'translation_{language}'])
    df_to_split = pd.read_csv(f'/content/drive/My Drive/DBDC_data/translation_files/DBDC_{sample}_translated_original_file_en.csv')

    for i in range(len(df_to_split.index)-1):
      parsed_dict = {}
      try:
        parsed_dict['utterance'] = df_to_split.iloc[i]['translation_en']
        parsed_dict['label'] = df_to_split.iloc[i]['label']
        parsed_dict[f'translation_{language}'] = model.translate(parsed_dict['utterance'], target_lang=language)

        df_translated = df_translated.append(parsed_dict, ignore_index=True, sort=False)
      except OSError:
        pass

    df_translated = df_translated.drop(['utterance'], axis=1)
    df_translated.to_csv(f'/content/drive/My Drive/DBDC_data/translation_files/DBDC_{sample}_translated_EasyNMT_file_{language}.csv', index=False)
    print('Translation of the language', language, 'is ready.')

def get_Google_translation(language, sample):
    """Get translation using Google translate."""
    df_to_split = pd.read_csv(f'/content/drive/My Drive/DBDC_data/translation_files/DBDC_{sample}_translated_original_file_en.csv')

    try:      
      df_to_split[f'translation_{language}'] = df_to_split.parallel_apply(lambda row: get_official_google_translate(row['translation_en'], 'en', language), axis=1)
      globals()[f'df_{sample}_translated_google_{language}'] = df_to_split.copy()
    except OSError:
      pass

    globals()[f'df_{sample}_translated_google_{language}'] = globals()[f'df_{sample}_translated_google_{language}'].drop(['translation_en'], axis=1)
    globals()[f'df_{sample}_translated_google_{language}'].to_csv(f'DBDC_{sample}_translated_Google_file_{language}.csv', index=False)
