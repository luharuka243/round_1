import os
import re
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from GdriveModels import download_google_drive_folder
from config import category_names_to_category, category_to_sub_category,master_mapper
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

if not os.path.exists('models_gdrive'):
    download_google_drive_folder()

class CyberCrimeDataset(Dataset):
    """Dataset class for cyber crime text classification"""
    def __init__(self, texts: List[str], 
                 categories: List[str], sub_categories: List[str]):
        self.texts = texts
        self.categories = categories
        self.sub_categories = sub_categories
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'category': self.categories[idx],
            'sub_category': self.sub_categories[idx]
        }

def custom_collate(batch):
    """Custom collate function to handle batch processing"""
    return {
        'text': [item['text'] for item in batch],
        'category': [item['category'] for item in batch],
        'sub_category': [item['sub_category'] for item in batch]
    }

def evaluate_predictions(y_true: List[str], y_pred: List[str], level_name: str):
    """
    Evaluate predictions for a given hierarchy level
    
    Args:
        y_true: List of true labels
        y_pred: List of predicted labels
        level_name: Name of the hierarchy level being evaluated
    """
    print(f"\n=== {level_name} Metrics ===")
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Generate and print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return accuracy

def process_batch(batch: Dict, encoder, models: Dict, selectors: Dict, 
                 label_encoders: Dict, category_to_sub_category: Dict,
                 master_mapper: Dict) -> Dict[str, List[str]]:
    """
    Process a batch of texts through the model hierarchy
    
    Args:
        batch: Dictionary containing batch data
        encoder: Sentence transformer encoder
        models: Dictionary of trained models
        selectors: Dictionary of feature selectors
        label_encoders: Dictionary of label encoders
        category_to_sub_category: Mapping of categories to subcategories
        master_mapper: Master mapping dictionary
    
    Returns:
        Dictionary containing predictions for all hierarchy levels
    """
    texts = [str(text).lower() for text in batch['text']]
    
    batch_results = {
        'pred_category_names': [],
        'pred_category': [],
        'pred_sub_category': [],
        'pred_sub_category_names': []
    }
    
    try:
        # Encode all texts in batch
        text_embeddings = encoder.encode(texts, show_progress_bar=False)
        text_embeddings = text_embeddings.reshape(len(texts), -1)
        
        # Predict main categories
        main_features = selectors['category_names'].transform(text_embeddings)
        main_cat_pred = models['category_names'].predict(main_features)
        main_categories = label_encoders['category_names'].inverse_transform(main_cat_pred)
        
        # Process each text in batch
        for idx, category_names in enumerate(main_categories):
            batch_results['pred_category_names'].append(category_names)
            
            try:
                # Predict category
                category_model_key = f'category_{category_names.replace(" ", "_").replace("/", "_").replace("&", "and")}'
                single_embedding = text_embeddings[idx:idx+1]
                
                category_features = selectors[category_model_key].transform(single_embedding)
                cat_pred = models[category_model_key].predict(category_features)
                category = label_encoders[category_model_key].inverse_transform(cat_pred)[0]
                
                # Predict subcategory
                if category in category_to_sub_category and len(category_to_sub_category[category]) > 1:
                    sub_category_names_model_key = f'sub_category_names_{category.replace(" ", "_").replace("/", "_").replace("&", "and")}'
                    sub_features = selectors[sub_category_names_model_key].transform(single_embedding)
                    mapped_sub_cat_pred = models[sub_category_names_model_key].predict(sub_features)
                    sub_category_names = label_encoders[sub_category_names_model_key].inverse_transform(mapped_sub_cat_pred)[0]
                    sub_category = find_immediate_key(master_mapper, sub_category_names)
                else:
                    sub_category_names = category_to_sub_category[category][0]
                    sub_category = find_immediate_key(master_mapper, sub_category_names)
                
            except KeyError as e:
                print(f"Warning: Model not found for prediction chain: {e}")
                category = "unknown"
                sub_category_names = "unknown"
                sub_category = "unknown"
            
            batch_results['pred_category'].append(category)
            batch_results['pred_sub_category'].append(sub_category)
            batch_results['pred_sub_category_names'].append(sub_category_names)
    
    except Exception as e:
        print(f"Error in processing batch: {str(e)}")
        # Fill with unknowns for this batch
        batch_size = len(texts)
        for key in batch_results:
            batch_results[key].extend(['unknown'] * batch_size)
    
    return batch_results

def run_inference_pipeline(test_df: pd.DataFrame, 
                         encoder, models: Dict, 
                         selectors: Dict, 
                         label_encoders: Dict,
                         category_to_sub_category: Dict,
                         master_mapper: Dict,
                         batch_size: int = 64) -> pd.DataFrame:
    """
    Run the complete inference pipeline
    
    Args:
        test_df: DataFrame containing test data
        encoder: Sentence transformer encoder
        models: Dictionary of trained models
        selectors: Dictionary of feature selectors
        label_encoders: Dictionary of label encoders
        category_to_sub_category: Mapping of categories to subcategories
        master_mapper: Master mapping dictionary
        batch_size: Batch size for processing
        
    Returns:
        DataFrame containing all predictions and metrics
    """
    # Create dataset and dataloader
    dataset = CyberCrimeDataset(
        texts=process_text_detailed(list(test_df['content_processed'])),
        categories=test_df['category'].apply(lambda x: clean_string(x)).tolist(),
        sub_categories=test_df['sub_category'].apply(lambda x: clean_string(x)).tolist()
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0
    )

    # Initialize results dictionary
    results = {
        'pred_category_names': [],
        'true_category': [],
        'pred_category': [],
        'true_sub_category': [],
        'pred_sub_category': [],
        'pred_sub_category_names': []
    }

    # Process batches
    print("\nProcessing batches...")
    total_batches = len(dataloader)

    with tqdm(total=total_batches, desc="Processing") as pbar:
        for batch_idx, batch in enumerate(dataloader):
            # Get predictions for batch
            batch_predictions = process_batch(
                batch, encoder, models, selectors, 
                label_encoders, category_to_sub_category, 
                master_mapper
            )
            
            # Store results
            # if 'category_names' in batch:
                # results['true_category_names'].extend(batch['category_names'])
            results['pred_category_names'].extend(batch_predictions['pred_category_names'])
            results['true_category'].extend(batch['category'])
            results['pred_category'].extend(batch_predictions['pred_category']) #
            # results['true_sub_category_names'].extend(batch['sub_category_names'])
            results['pred_sub_category_names'].extend(batch_predictions['pred_sub_category_names']) 
            results['true_sub_category'].extend(batch['sub_category'])
            results['pred_sub_category'].extend(batch_predictions['pred_sub_category']) #
            # Update progress and show intermediate metrics
            pbar.update(1)
            if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                show_intermediate_metrics(results, batch_idx, total_batches, pbar)

    # Convert results to DataFrame and calculate final metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv('prediction_results.csv', index=False)
    
    print("\nCalculating final metrics...")
    calculate_final_metrics(results_df)
    
    return results_df

def show_intermediate_metrics(results: Dict, batch_idx: int, 
                            total_batches: int, pbar: tqdm):
    """Show intermediate metrics during batch processing"""
    pbar.write(f"\nBatch {batch_idx + 1}/{total_batches}")
    
    for level in ['category', 'category_names', 'sub_category_names', 'sub_category']:
        true_key = f'true_{level}'
        pred_key = f'pred_{level}'
        if true_key in results and len(results[true_key]) > 0:
            current_accuracy = accuracy_score(
                results[true_key][:len(results[pred_key])],
                results[pred_key]
            )
            pbar.write(f"Current {level} Accuracy: {current_accuracy:.4f}")

def clean_string(s):
    if not isinstance(s, str):
        return s
    
    # Replace spaces with underscore
    s=s.lower()
    s = s.replace(' ', '_')
    
    # Replace special characters with asterisk
    s = re.sub(r'[^a-zA-Z0-9_.]', '*', s)
    s = s.replace(".","")
    
    # Remove consecutive special characters
    s = re.sub(r'[*_]+', lambda m: '_' if '_' in m.group() else '_', s)
    
    return s

def clean_json_mapping(json_mapping):
    """
    Cleans a JSON mapping by replacing special characters and spaces with * or _,
    and removing consecutive special characters.
    
    Args:
        json_mapping (dict): Input JSON mapping to clean
        
    Returns:
        dict: Cleaned JSON mapping
    """
    def clean_string(s):
        if not isinstance(s, str):
            return s
        
        # Replace spaces with underscore
        s = s.replace(' ', '_')
        
        # Replace special characters with asterisk
        s = re.sub(r'[^a-zA-Z0-9_.]', '*', s)
        s = s.replace(".","")
        
        # Remove consecutive special characters
        s = re.sub(r'[*_]+', lambda m: '_' if '_' in m.group() else '_', s)
        
        return s
    
    def process_value(value):
        if isinstance(value, dict):
            return {clean_string(k): process_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [clean_string(item) for item in value]
        else:
            return clean_string(value)
    
    return process_value(json_mapping)

def calculate_final_metrics(results_df: pd.DataFrame):
    """Calculate and display final metrics for all hierarchy levels"""
    if 'true_category_names' in results_df.columns and results_df['true_category_names'].iloc[0] != 'unknown':
        evaluate_predictions(
            results_df['true_category_names'],
            results_df['pred_category_names'],
            'Main Category'
        )

    evaluate_predictions(
        results_df['true_category'],
        results_df['pred_category'],
        'category'
    )


    evaluate_predictions(
        results_df['true_sub_category'],
        results_df['pred_sub_category'],
        'Sub-Category'
    )

    
# Mappings
category_names_to_category = clean_json_mapping(category_names_to_category)
category_to_sub_category = clean_json_mapping(category_to_sub_category)

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def process_text_detailed(contents):
    # Ensure contents is a list
    input_was_string = isinstance(contents, str)
    if isinstance(contents, str):
        contents = [contents]
    
    # Define a comprehensive URL regex pattern
    url_pattern = r'(https?://\S+|www\.\S+|\b\w+\.\w+\.\S+)'
    
    # Process each content string
    processed_contents = []
    for text in contents:
        text= str(text)
        
        # Check for URLs
        if re.search(url_pattern, text, re.IGNORECASE):
            # Clean URLs
            text = re.sub(r'[^\w\s.]', ' ', text).lower().strip()
        else:
            # 1. Convert to lower
            text = text.lower()
            
            # 4. Replace special characters with spaces
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            
            # 5. Convert multiple spaces to single space
            text = re.sub(r'\s+', ' ', text).strip()
            
            # 6. Remove consecutive repeated letters (3 or more)
            text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # 7. Remove if less than 3 words
        if len(text.split())==0:
            text=' '
        
        # 8. Remove if only numbers
        if text.replace(' ', '').isnumeric():
            text=' '
        
        processed_contents.append(text)
    # Return single string if input was single string, otherwise return list

    if len(processed_contents)>0:
        processed_result = processed_contents[0] if input_was_string else processed_contents
    else :
        processed_result=''


    return processed_result

def load_models(models_path='models_gdrive/models/') -> Tuple[object, Dict, Dict, Dict]:
    """
    Load all saved models, encoders, and selectors from the specified path
    
    Args:
        models_path (str): Path to directory containing saved models
        
    Returns:
        Tuple containing:
        - sentence encoder
        - dictionary of trained models
        - dictionary of label encoders
        - dictionary of feature selectors
    """
    print("Loading models...")
    models = {}
    label_encoders = {}
    selectors = {}
    
    # Load the same sentence transformer used in training
    encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    
    # Load main category model components
    try:
        models['category_names'] = joblib.load(os.path.join(models_path, 'category_names_model.joblib'))
        label_encoders['category_names'] = joblib.load(os.path.join(models_path, 'category_names_encoder.joblib'))
        selectors['category_names'] = joblib.load(os.path.join(models_path, 'category_names_selector.joblib'))
    except Exception as e:
        raise RuntimeError(f"Failed to load main category model components: {str(e)}")
    
    # Load all category and subcategory models
    model_files = list(Path(models_path).glob('*_model.joblib'))
    for file in tqdm(model_files, desc="Loading models"):
        if file.name != 'category_names_model.joblib':
            key = file.name.replace('_model.joblib', '')
            try:
                models[key] = joblib.load(file)
                label_encoders[key] = joblib.load(str(file).replace('_model.joblib', '_encoder.joblib'))
                selectors[key] = joblib.load(str(file).replace('_model.joblib', '_selector.joblib'))
            except Exception as e:
                print(f"Warning: Failed to load model components for {key}: {str(e)}")
    
    return encoder, models, label_encoders, selectors

def predict_single(text: str, encoder, models: Dict, selectors: Dict, 
                  label_encoders: Dict, category_to_sub_category: Dict,
                  master_mapper: Dict) -> Dict[str, str]:
    """
    Process a single text through the hierarchical model chain
    
    Args:
        text (str): Input text to classify
        encoder: Sentence transformer encoder
        models (Dict): Dictionary of trained models
        selectors (Dict): Dictionary of feature selectors
        label_encoders (Dict): Dictionary of label encoders
        category_to_sub_category (Dict): Mapping from categories to subcategories
        master_mapper (Dict): Master mapping of categories
        
    Returns:
        Dict containing predictions for category_names, category, sub_category, 
        and sub_category_names
    """
    try:
        start=time.time()
        # Preprocess and encode text
        processed_text = process_text_detailed(text)
        if not processed_text:
            return {
                'pred_category_names': 'any_other_cyber_crime',
                'pred_retagged_category': 'any_other_cyber_crime',
                'pred_retagged_sub_category': 'other',
                'pred_sub_category_names': 'other'
            }
        text_embedding = encoder.encode([processed_text], show_progress_bar=False)
        text_embedding = text_embedding.reshape(1, -1)
        
        # Predict main category
        main_features = selectors['category_names'].transform(text_embedding)
        main_cat_pred = models['category_names'].predict(main_features)
        category_names = label_encoders['category_names'].inverse_transform(main_cat_pred)[0]
        
        # Predict category based on main category
        category_model_key = f'category_{category_names.replace(" ", "_").replace("/", "_").replace("&", "and")}'
        try:
            category_features = selectors[category_model_key].transform(text_embedding)
            cat_pred = models[category_model_key].predict(category_features)
            category = label_encoders[category_model_key].inverse_transform(cat_pred)[0]
        except KeyError:
            print(f"Warning: No category model found for {category_names}")
            return {
                'pred_category_names': category_names,
                'pred_retagged_category': 'unknown',
                'pred_retagged_sub_category': 'unknown',
                'pred_sub_category_names': 'unknown'
            }
        
        # Predict subcategory if multiple options exist
        if category in category_to_sub_category and len(category_to_sub_category[category]) > 1:
            sub_category_names_model_key = f'sub_category_names_{category.replace(" ", "_").replace("/", "_").replace("&", "and")}'
            try:
                sub_features = selectors[sub_category_names_model_key].transform(text_embedding)
                mapped_sub_cat_pred = models[sub_category_names_model_key].predict(sub_features)
                sub_category_names = label_encoders[sub_category_names_model_key].inverse_transform(mapped_sub_cat_pred)[0]
                sub_category = find_immediate_key(master_mapper, sub_category_names)
            except KeyError:
                print(f"Warning: No subcategory model found for {category}")
                sub_category_names = category_to_sub_category[category][0]
                sub_category = find_immediate_key(master_mapper, sub_category_names)
        else:
            sub_category_names = category_to_sub_category[category][0]
            sub_category = find_immediate_key(master_mapper, sub_category_names)
        
        return {
            'pred_category_names': category_names,
            'pred_retagged_category': category,
            'pred_retagged_sub_category': sub_category,
            'pred_sub_category_names': sub_category_names
        }
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {
            'pred_category_names': 'unknown',
            'pred_retagged_category': 'unknown',
            'pred_retagged_sub_category': 'unknown',
            'pred_sub_category_names': 'unknown'
        }

def find_immediate_key(dictionary, search_value):
    """
    Find the immediate key for a given value in a nested dictionary.
    
    Args:
    dictionary (dict): The nested dictionary to search
    search_value (str): The value to find
    
    Returns:
    str or None: The immediate key if found, None otherwise
    """
    for outer_key, inner_dict in dictionary.items():
        for inner_key, values in inner_dict.items():
            if search_value in values:
                return inner_key
    return None

def save_detailed_results(results_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save detailed analysis of the results"""
    # Combine original text with predictions
    detailed_results = pd.concat([
        test_df['content_processed'],
        results_df
    ], axis=1)
    
    # Add correctness columns
    # detailed_results['category_names_correct'] = (
    #     detailed_results['true_category_names'] == 
    #     detailed_results['pred_category_names']
    # )
    detailed_results['category_correct'] = (
        detailed_results['true_category'] == 
        detailed_results['pred_category']
    )
    detailed_results['sub_category_correct'] = (
        detailed_results['true_sub_category'] == 
        detailed_results['pred_sub_category']
    )
    
    # Save to CSV
    detailed_results.to_csv('detailed_prediction_results.csv', index=False)
    
    # Save error analysis
    error_cases = detailed_results[
        ~(detailed_results['category_correct'] & 
          detailed_results['sub_category_correct'])
    ]
    error_cases.to_csv('prediction_errors.csv', index=False)

def analyze_examples(results_df: pd.DataFrame, test_df: pd.DataFrame, n_examples: int = 5):
    """Analyze specific examples from the results"""
    print("\n=== Example Predictions ===")
    
    # Sample some random examples
    indices = np.random.choice(len(results_df), min(n_examples, len(results_df)), replace=False)
    
    for idx in indices:
        print("\nText:")
        print(test_df['content_processed'].iloc[idx][:200] + "...")  # Show first 200 chars
        
        print("\nPredictions:")
        print(f"Main Category: {results_df['pred_category_names'].iloc[idx]} "
              f"(True: {results_df['true_category_names'].iloc[idx]})")
        print(f"Category: {results_df['pred_retagged_category'].iloc[idx]} "
              f"(True: {results_df['true_retagged_category'].iloc[idx]})")
        print(f"Sub-Category: {results_df['pred_retagged_sub_category'].iloc[idx]} "
              f"(True: {results_df['true_retagged_sub_category'].iloc[idx]})")
        print("-" * 80)



# Example usage with your mapping
master_mapper = clean_json_mapping(master_mapper)
# Load models and vectorizer
encoder, models, label_encoders, selectors = load_models(models_path='models_gdrive/models/')


'''
TO PUT CSV
'''
# Put CSV path here to predict on complete csv
PATH_CSV="data/test.csv"
print("Loading test data...")
test_df = pd.read_csv('data/test.csv')
# test_df=test_df.head(100)
test_df = test_df.dropna(subset=['crimeaditionalinfo', 'category'], how='all')
test_df['sub_category'] = test_df['sub_category'].fillna(test_df['category'])
# Clean up the text data
test_df['content_processed'] = test_df['crimeaditionalinfo'].fillna('')
test_df['content_processed'] = test_df['content_processed'].astype(str)

results_df = run_inference_pipeline(
    test_df=test_df,
    encoder=encoder,
    models=models,
    selectors=selectors,
    label_encoders=label_encoders,
    category_to_sub_category=category_to_sub_category,
    master_mapper=master_mapper,
    batch_size=64
)
save_detailed_results(results_df, test_df)

'''
FOR SINGLE TEXT STRING
'''
# start=time.time()
# results_df = predict_single(
#     text="hi i am a cyber crime, there is lot of problem in cyber crime in upi",
#     encoder=encoder,
#     models=models,
#     selectors=selectors,
#     label_encoders=label_encoders,
#     category_to_sub_category=category_to_sub_category,
#     master_mapper=master_mapper
# )
# print(f"Time taken for single text : {time.time()-start}")

# print(results_df)
