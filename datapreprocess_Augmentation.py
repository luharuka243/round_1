
from nlpaug.util.file.download import DownloadUtil
DownloadUtil.download_word2vec(dest_dir='.') # Download word2vec model
DownloadUtil.download_glove(model_name='glove.6B', dest_dir='.') # Download GloVe model
DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='.') # Download fasttext model


import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.char as nac
import pandas as pd
import random
import os
os.environ["TOKENIZERS_PARALLELISM"]="true"
from joblib import Parallel, delayed


def flatten_list(mixed_list):
    """
    Flattens a list containing strings or lists of strings.
    
    Args:
        mixed_list (list): A list containing strings and lists of strings.

    Returns:
        list: A flattened list of strings.
    """
    flattened = []
    for item in mixed_list:
        if isinstance(item, list):  # Check if the item is a list
            flattened.extend(item)  # Extend the flattened list with the inner list
        elif isinstance(item, str):  # Check if the item is a string
            flattened.append(item)  # Add the string to the flattened list
    return flattened

class AdvancedTextAugmenter:
    def __init__(self, languages=None):
        """
        Initialize augmenters with optional language support
        
        Args:
            languages (list): List of language codes for translation augmentation
        """
        self.languages = languages or ['fr', 'de', 'es']
        
        # Word-level augmenters
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
        self.word_embedding_aug = naw.WordEmbsAug(
            model_type='word2vec', 
            model_path='GoogleNews-vectors-negative300.bin'  # Optional: specify a pre-trained model
        )
        
        # Contextual augmenters (requires transformer models)
        try:
            self.bert_aug = naw.ContextualWordEmbsAug(
                model_path='xlm-roberta-base', 
                action='substitute',device="cuda"
            )
        except Exception as e:
            print(f"XLM Roberta augmentation not available: {e}")
            self.bert_aug = None
        
        # Back-translation augmenters
        self.back_translation_augs = [
            naw.BackTranslationAug(
                from_model_name=f'Helsinki-NLP/opus-mt-en-{lang}',
                to_model_name=f'Helsinki-NLP/opus-mt-{lang}-en',device="cuda"
            ) for lang in self.languages
        ]
        
        # Sentence-level augmenter
        self.sentence_aug = nas.RandomSentAug()

    def augment_text(self, text, num_augmentations=5, augmentation_techniques=None, categories=None):
        """
        Apply multiple augmentation techniques to the input text
        
        Args:
            text (str): Input text to augment
            num_augmentations (int): Number of augmentations to generate
            augmentation_techniques (list): Specific techniques to use
        
        Returns:
            list: Augmented text variations
        """
        # Default augmentation techniques if not specified
        if augmentation_techniques is None:
            augmentation_techniques = [
                'synonym',
                'word_embedding',
                'back_translation',
                'contextual',
                'sentence_swap'
            ]
        augmented_texts = []
        
        # Synonym Replacement
        print(f"Generating the data for categories {categories} using synonym")
        if 'synonym' in augmentation_techniques:
            augmented_texts.extend(
                [self.synonym_aug.augment(text) for _ in range(num_augmentations//2)]
            )
        
        # Word Embedding Augmentation
        print(f"Generating the data for categories {categories} using word_embedding")
        if 'word_embedding' in augmentation_techniques:
            try:
                augmented_texts.extend(
                    [self.word_embedding_aug.augment(text) for _ in range(num_augmentations//2)]
                )
            except Exception as e:
                print(f"Word embedding augmentation failed: {e}")
        
        # Back Translation
        print(f"Generating the data for categories {categories} using back_translation")
        if 'back_translation' in augmentation_techniques:
            for translator in self.back_translation_augs:
                try:
                    augmented_texts.append(translator.augment(text)[0])
                except Exception as e:
                    print(f"Back translation augmentation failed: {e}")
        
        # Contextual Word Embeddings
        print(f"Generating the data for categories {categories} by change contextual meaning")
        if 'contextual' in augmentation_techniques and self.bert_aug:
            augmented_texts.extend(
                [self.bert_aug.augment(text) for _ in range(num_augmentations//2)]
            )
        
        # Sentence-level Augmentation
        print(f"Generating the data for categories {categories} using sentence_swapping")
        if 'sentence_swap' in augmentation_techniques:
            augmented_texts.append(self.sentence_aug.augment(text)[0])
            
            
        augmented_texts=flatten_list(augmented_texts)
        # Remove duplicates and limit to unique augmentations
        unique_augmented_texts = list(set(augmented_texts))
        
        print(f"Total unique generated data text for {categories} is {len(unique_augmented_texts)}")
        return unique_augmented_texts

text_augmenter = AdvancedTextAugmenter()



def augment_original_mapped_sub_category(df, per_sample_target_count=1000, text_columns=None):
    """
    Augment rows for original_mapped_sub_category with fewer than target_count instances
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_count (int): Target number of instances per category
        text_columns (list): Columns to apply text augmentation
        output_file (str): Path to save augmented data CSV
    
    Returns:
        pd.DataFrame: Augmented DataFrame
    """
    # Default to all object columns if not specified
    if text_columns is None:
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # Identify categories with low counts
    category_counts = df['mapped_sub_category'].value_counts()
    low_count_categories = category_counts[category_counts < per_sample_target_count].index.tolist()
    
    # Augmented samples storage
    augmented_samples = []
    
    print("Categories to be augmented:")
    for category in low_count_categories:
        print(f"{category}: {category_counts[category]}")
        
    # remove this one while submission
    done=['online job fraud','dematdepository fraud','ransomware attack','tampering with computer source documents',
         "unauthorized social media access"
         ]
    low_count_categories=list(set(low_count_categories)-set(done))
    
    # Augmentation for each low-count category
    for category in low_count_categories:
        print("*"*75)
        print(f"Generation starts for category: {category}")
        
        # Get subset of current category
        category_subset = df[df['mapped_sub_category'] == category]
        current_count = len(category_subset)
        needed_count = per_sample_target_count - current_count
        augmented_sample=[]
        base_sample_text = category_subset.sample(n=1).crimeaditionalinfo.to_list()
        augmented_variations = text_augmenter.augment_text(
                    base_sample_text,
                    num_augmentations=needed_count,
                    categories=category
                )
        augmented_variations_df=pd.DataFrame({
            "content":augmented_variations,
            "mapped_sub_category":[category for _ in range(len(augmented_variations))]
        })
        if "/" in category:
            category = category.replace("/", "_")
            augmented_variations_df.to_csv(f"augumented_data/augmented-data-{category}.csv")
        else:
            augmented_variations_df.to_csv(f"augumented_data/augmented-data-{category}.csv")
        augmented_samples.append(augmented_variations_df)
    
    augmented_df = pd.concat(augmented_samples,ignore_index=True)
    
    # Save augmented data to CSV
    augmented_df.to_csv(f"data/augmented-data-all.csv", index=False)
    print(f"\nAugmented data saved to data/augmented-data-all.csv")
    return augmented_df


data = pd.read_csv("data/train_data_less_than_1000_count.csv",index_col="Unnamed: 0")
data.dropna(inplace=True)

# Augment the dataset
augmented_df = augment_original_mapped_sub_category(data,text_columns="crimeaditionalinfo")

print(f"Augemented data is {augmented_df}")