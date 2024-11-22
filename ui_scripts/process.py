import re

def process_text(contents):
    # Ensure contents is a list
    input_was_string = isinstance(contents, str)
    print("input is string:" , input_was_string)
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
    print(f"Processed contents : {processed_contents}")
    # Return single string if input was single string, otherwise return list

    if len(processed_contents)>0:
        processed_result = processed_contents[0] if input_was_string else processed_contents
    else :
        processed_result=''


    return processed_result

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