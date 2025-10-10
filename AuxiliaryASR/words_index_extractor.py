import os
from collections import Counter
import re
import yaml
import click
# from nltk.tokenize import word_tokenize
# from phonemizer import phonemize
# from phonemizer.separator import Separator

@click.command()
@click.option('-p', '--config_path', default='./Configs/config.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))
    TRAINING_FILE = config["train_data"]
    VALIDATION_FILE = config["val_data"]
    OOD_FILE = config["ood_data"]
    
    # Step 1: Load all sentences
    with open(TRAINING_FILE, "r", encoding="utf-8") as f:
        sentences_training = [line.strip().split("|")[1] for line in f if "|" in line]
    with open(VALIDATION_FILE, "r", encoding="utf-8") as f:
        sentences_validation = [line.strip().split("|")[1] for line in f if "|" in line]
    with open(OOD_FILE, "r", encoding="utf-8") as f:
        sentences_ood = [line.strip().split("|")[1] for line in f if "|" in line]
    
    all_sentences = sentences_training + sentences_validation + sentences_ood
    
    # Set separator to mark phones
    # separator = Separator(phone='|', word='', syllable='')
    
    # Step 2: Phonemize and save phonemes
    word_index_dict = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3, " ": 4}
    i = 4
    for sentence in all_sentences:
        # sentence = sentence.replace('-', ' ').replace("(", '"').replace(")", '"')
        # ps = phonemize(
        #     [sentence],
        #     language='en-us',  # or 'cs', 'sk', etc. — you can switch dynamically
        #     backend='espeak',
        #     strip=True,
        #     with_stress=False,
        #     language_switch='remove-flags',
        #     separator=separator,
        #     preserve_punctuation=False,
        #     njobs=1
        # )
        phonemes = list("".join(sentence.split(' ')))
        for phoneme in phonemes:
            if phoneme == '"':
                phoneme = '""' # escape the quote sign, so that our CSV format remains valid
    
            if not phoneme in word_index_dict:
                i += 1
                word_index_dict[phoneme] = i
    
    # Also get phonemes from OOD text, which is already
    
    # Step 3: Count and collect unique phonemes
    #phoneme_counter = Counter(phoneme_list)
    
    # Special tokens (like padding, start-of-sequence, end-of-sequence)
    #special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", " "]
    #phoneme_counter.update(special_tokens)
    # for token in special_tokens:
    #     phoneme_counter[token] = 1  # force inclusion
    
    # Step 4: Create word index dictionary (mapping phonemes to unique indices)
    # word_index_dict = {phoneme: idx for idx, (phoneme, _) in enumerate(phoneme_counter.most_common(), 1)}
    # word_index_dict["<PAD>"] = 0
    # word_index_dict["<EOS>"] = len(word_index_dict)
    # word_index_dict = {"<PAD>": 0}
    # for idx, (phoneme, _) in enumerate(phoneme_counter.most_common(), start=1):
    #     if phoneme != "<PAD>":
    #         word_index_dict[phoneme] = idx
    
    # Ensure <EOS> is last
    
    # special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>", " "]
    # word_index_dict = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3, " ": 4}
    # counter = len(word_index_dict)
    # for idx, (phoneme, _) in enumerate(phoneme_counter.most_common()):
    #     phoneme = phoneme.strip()
    #     if phoneme not in special_tokens and phoneme not in word_index_dict:
    #         if phoneme == '"':
    #             phoneme = '""' # escape the quote sign, so that our CSV format remains valid
    #         word_index_dict[phoneme] = counter
    #         counter += 1
    
    # Step 5: Save dictionary to file
    #output_file = "word_index_dict_en_test.txt"
    output_file = "word_index_dict.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for phoneme, idx in word_index_dict.items():
            f.write(f'"{phoneme}",{idx}\n')
    
    #print(f"✅ Saved word_index_dict.txt with {len(word_index_dict)} phonemes.")
    print(f"✅ Saved {output_file} with {len(word_index_dict)} entries.")

if __name__=="__main__":
    main()
