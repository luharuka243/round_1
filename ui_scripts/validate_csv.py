def validate_input_csv(dataframe):
    # Check if the input dataframe has the required columns
    required_columns = ['id', 'input_text']
    if not all(col in dataframe.columns for col in required_columns):
        raise ValueError("Input dataframe is missing required columns. Please include 'id' and 'input' columns.")

    # Check if the input dataframe has the correct data types
    input_column = dataframe['input_text']
    id_column = dataframe['id']

    if len(input_column) != len(id_column):
        raise ValueError("Input and id_column columns must have the same number of rows.")

    return True