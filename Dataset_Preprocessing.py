from datasets import Audio, load_dataset,DatasetDict
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
#import hazm
import os
import string
import re
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import time
from tqdm import tqdm
import sys
from datasets import Audio
import csv
from jiwer import cer as jiwer_cer
import numpy as np

# Define preprocessing functions
def preprocess_dataset(dataset):
    common_voice =dataset
    print(dataset)
    
    chars_to_ignore = [
        ",", "?", ".", "!", "-", ";", ":", '""', "%", "'", '"', "�",
        "#", "!", "؟", "?", "«", "»", "،", "(", ")", "؛", "'ٔ", "٬",'ٔ', ",", "?", 
        ".", "!", "-", ";", ":",'"',"“", "%", "‘", "”", "=", "–", "…", "_", "”", '“', '„',
        'ā', 'š',
    ]
    # In case of farsi
    chars_to_ignore = chars_to_ignore + list(string.ascii_lowercase + string.digits)
    chars_to_mapping = {
        'ك': 'ک', 'دِ': 'د', 'بِ': 'ب', 'زِ': 'ز', 'ذِ': 'ذ', 'شِ': 'ش', 'سِ': 'س', 'ى': 'ی',
        'ي': 'ی', 'أ': 'ا', 'ؤ': 'و', "ے": "ی", "ۀ": "ه", "ﭘ": "پ", "ﮐ": "ک", "ﯽ": "ی",
        "ﺎ": "ا", "ﺑ": "ب", "ﺘ": "ت", "ﺧ": "خ", "ﺩ": "د", "ﺱ": "س", "ﻀ": "ض", "ﻌ": "ع",
        "ﻟ": "ل", "ﻡ": "م", "ﻢ": "م", "ﻪ": "ه", "ﻮ": "و", 'ﺍ': "ا", 'ة': "ه",
        'ﯾ': "ی", 'ﯿ': "ی", 'ﺒ': "ب", 'ﺖ': "ت", 'ﺪ': "د", 'ﺮ': "ر", 'ﺴ': "س", 'ﺷ': "ش",
        'ﺸ': "ش", 'ﻋ': "ع", 'ﻤ': "م", 'ﻥ': "ن", 'ﻧ': "ن", 'ﻭ': "و", 'ﺭ': "ر", "ﮔ": "گ",
            
        # "ها": "  ها", "ئ": "ی",
        "۱۴ام": "۱۴ ام",
            
        "a": " ای ", "b": " بی ", "c": " سی ", "d": " دی ", "e": " ایی ", "f": " اف ",
        "g": " جی ", "h": " اچ ", "i": " آی ", "j": " جی ", "k": " کی ", "l": " ال ",
        "m": " ام ", "n": " ان ", "o": " او ", "p": " پی ", "q": " کیو ", "r": " آر ",
        "s": " اس ", "t": " تی ", "u": " یو ", "v": " وی ", "w": " دبلیو ", "x": " اکس ",
        "y": " وای ", "z": " زد ",
        "\u200c": " ", "\u200d": " ", "\u200e": " ", "\u200f": " ", "\ufeff": " ",
    }
    def multiple_replace(text, chars_to_mapping):
        pattern = "|".join(map(re.escape, chars_to_mapping.keys()))
        return re.sub(pattern, lambda m: chars_to_mapping[m.group()], str(text))

    def remove_special_characters(text, chars_to_ignore_regex):
        text = re.sub(chars_to_ignore_regex, '', text).lower() + " "
        return text

    def Remove_punctuation_and_covert_finglish(batch, chars_to_ignore=chars_to_ignore, chars_to_mapping=chars_to_mapping):
        if batch is None:
            print("Warning: Batch is None")
            return None

        if 'sentence' not in batch or batch['sentence'] is None:
            print("Warning: 'sentence' key not found or value is None in batch")
            print(batch['path'])
            return None
        
        chars_to_ignore_regex = f"""[{"".join(chars_to_ignore)}]"""
        text = batch['sentence'].strip()
        
        if text == '':
            #print("Warning: 'sentence' value is an empty string in batch")
            # Handle empty strings by returning an appropriate placeholder value
            batch['sentence'] = "EMPTY_SENTENCE"
            return batch
        
        text = multiple_replace(text, chars_to_mapping)
        text = remove_special_characters(text, chars_to_ignore_regex)
        text = re.sub(" +", " ", text)
        
        if text is None or not len(text) > 0:
            print("Warning: Text after preprocessing is None or empty")
            return None
        
        _text = []
        for word in text.split():
            try:
                word = int(word)
                _text.append(words(word))
            except:
                _text.append(word)
                
        text = " ".join(_text) + " "
        text = text.strip()
        
        if text == '':
            print("Warning: Text after tokenization became an empty string")
            return None
        
        batch['sentence'] = text   
        return batch
    
    def preprocess_text(text):
        # Define regular expression pattern to identify "ها", "های", or "هایی" at the end of words
        pattern = r'\b(\S+)(?:\s+(ها|های| هایی|ام|اش|ای|اید|اند))\b'

        # Define a function to concatenate the suffix with the previous word
        def replace(match):
            return match.group(1) + match.group(2)

        # Apply the regex substitution to the text
        processed_text = re.sub(pattern, replace, text)

        return processed_text

    chars_to_remove_regex = '[\،\?\.\!\-\;\:\"\“\%\‘\”\�\']'
    mi_space_word_regex = r'(?<=\S)\s+می\s+(\S+)'

    def remove_special_characters_mi(text):
        # Concatenate 'می' with the next word if separated by one space from its previous word
        text = re.sub(mi_space_word_regex, r' می\1', text)
        # Remove other special characters
        return re.sub(chars_to_remove_regex, ' ', text).lower()
    
    def remove_extra_spaces(sentence):
    # Use regular expression to replace multiple spaces with a single space
        return re.sub(r'\s+', ' ', sentence)
    
    common_voice = common_voice.filter(lambda example: 'sentence' in example and example['sentence'] is not None)
    common_voice = common_voice.map(Remove_punctuation_and_covert_finglish, fn_kwargs={"chars_to_ignore": chars_to_ignore, "chars_to_mapping": chars_to_mapping})
    common_voice = common_voice.map(lambda example: {'sentence': preprocess_text(example['sentence'])})
    common_voice = common_voice.map(lambda example: {'sentence': remove_special_characters_mi(example['sentence'])}) 
    common_voice = common_voice.map(lambda example: {'sentence': remove_extra_spaces(example['sentence'])}) 

    preprocessed_dataset= common_voice
    return preprocessed_dataset

