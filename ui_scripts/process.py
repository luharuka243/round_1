import re

def process_text(contents):
    # Ensure contents is a list
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
        if len(text.split()) < 3:
            continue
        
        # 8. Remove if only numbers
        if text.replace(' ', '').isnumeric():
            continue
        
        processed_contents.append(text)
    
    # Return single string if input was single string, otherwise return list
    return processed_contents[0] if len(processed_contents) == 1 else processed_contents