def install_CAMeL_tools():
    pass

def installLibraries():
    pass

def push_file_to_github(access_token:str, local_file_path:str, github_repo_path:str, github_folder_path:str='', commit_message:str=None) -> dict:
    """
    Pushes a local file to a GitHub repository.

    Args:
    - access_token (str): A GitHub access token with write access to the repository.
    - local_file_path (str): The path to the local file that will be pushed to the GitHub repository.
    - github_repo_path (str): The path of the GitHub repository in the format 'owner/repo'.
    - github_folder_path (str): Optional. The path of the folder in the GitHub repository where the file will be pushed to. Defaults to the root directory of the repository.
    - commit_message (str): Optional. A custom commit message for the file push. If not specified, a default commit message will be used.

    Returns:
    - dict: The response JSON fromthe API call.
    
    Example Usage:
    push_file_to_github(
        access_token = 'YOUR_GITHUB_ACCESS_TOKEN',
        local_file_path = 'file.txt',
        github_repo_path = 'essawey/MarareProject',
        github_folder_path = '',
        commit_message="Testing a automatic function"
        )
    """
    import requests
    import base64
    import os

    # Build the API endpoint URL
    if github_folder_path:
        api_url = f'https://api.github.com/repos/{github_repo_path}/contents/{github_folder_path}/{os.path.basename(local_file_path)}'
    else:
        api_url = f'https://api.github.com/repos/{github_repo_path}/contents/{os.path.basename(local_file_path)}'

    # Read the content of the local file
    with open(local_file_path, 'rb') as file:
        file_content = file.read()

    # Encode the file content in Base64
    file_content_base64 = base64.b64encode(file_content).decode('utf-8')

    # Set the request headers
    headers = {
        'Authorization': f'token {access_token}',
        'Accept': 'application/vnd.github+json'
    }

    # Make a GET request to check if the file already exists
    response = requests.get(api_url, headers=headers)

    # If the file exists, extract the current SHA hash
    if response.status_code == 200:
        current_file_sha = response.json()['sha']
    
        # Set the request payload to delete the existing file
        delete_payload = {
            'message': commit_message if commit_message else f'Delete {os.path.basename(local_file_path)}',
            'sha': current_file_sha,
            'branch': 'main'
        }

        # Make a PUT request to delete the existing fileContinued:
        delete_response = requests.delete(api_url, headers=headers, json=delete_payload)

        # Print the response message and status code
        print(f'Delete response message: {delete_response.json()["message"]}')
        print(f'Delete status code: {delete_response.status_code}')
    
    # Set the request payload to create or update the file
    payload = {
        'message': commit_message if commit_message else f'Add {os.path.basename(local_file_path)}',
        'content': file_content_base64,
        'branch': 'main'
    }

    # Make the API request to create or update the file on GitHub
    response = requests.put(api_url, headers=headers, json=payload)

    # Print the response message and status code
    print(f'Push status code: {response.status_code}')

    # Return the response JSON
    return response.json()

def clean_English_Dict(text:str) -> str:
    """
    Cleans a given text by removing unwanted characters and converting all text to lowercase.
    
    Args:
        text (str): The input text to be cleaned.
    
    Returns:
        str: The cleaned text.
    """
    import re

    clean_text = re.sub('[^\w\s]', '', text)
    clean_text = re.sub('\s+', ' ', clean_text)
    clean_text = clean_text.lower() 
    
    return clean_text

def removePunctuation(text:str) -> str:
    """
    Remove punctuation from the given text and return the modified text.

    Args:
        text (str): A string of text that may contain punctuation.

    Returns:
        str: The modified string with punctuation removed.
    """
    import string

    # Define a string of Arabic and English punctuation to remove
    punctuations = "،؛؟ـ٪٫٬" + string.punctuation +  string.printable + string.whitespace

    for punctuation in punctuations:
        text = text.replace(punctuation, " ")

    return text

def removeTashkeel(text:str) -> str:
    """
    Remove diacritical marks (tashkeel) from the given text and return the cleaned text.

    Args:
        text (str): A string of text that may contain diacritical marks.

    Returns:
        str: The modified string with diacritical marks removed.
    """
    import re

    cleanText = [re.sub('[\u0617-\u061A\u064B-\u0652]', '', word) for word in text]

    # Join the list of words back into a single string and return it
    return ''.join(cleanText)

def removeStopWords(text:str) -> str:
    """
    Remove stop words from the given text and return the cleaned text.

    Args:
        text (str): A string of text that may contain stop words.

    Returns:
        str: The modified string with stop words removed.
    """
    from nltk.corpus import stopwords

    # Split the input text into a list of words
    text = text.split()

    # Load the set of Arabic stop words from the NLTK corpus
    stop_words = set(stopwords.words('arabic'))

    # Remove the stop words from the Set of words
    cleanText = [word for word in text if word not in stop_words]

    # Join the remaining words back into a single string
    return ' '.join(cleanText)

def stemming(text:str) -> str:
    """
    Perform stemming on the given Arabic text using the MLEDisambiguator from CamelTools and return the modified text.

    Args:
        text (str): A string of Arabic text to be stemmed.

    Returns:
        str: The modified string with stemming applied.
    """
    install_CAMeL_tools()    
    from camel_tools.disambig.mle import MLEDisambiguator

    # Split the input text into a list of words
    text = text.split()

    # Load the MLEDisambiguator model for Arabic from CamelTools
    mle = MLEDisambiguator.pretrained()

    # Disambiguate the list of words using the MLEDisambiguator model
    # The disambiguated words are returned as a list of DisambiguatedData objects
    disambig = mle.disambiguate(text)

    # For each disambiguated word d in disambig, d.analyses is a list of analyses
    # sorted from most likely to least likely. Therefore, d.analyses[0] would
    # be the most likely analysis for a given word. Below we extract the word root, stemming
    root = [d.analyses[0].analysis['lex'] for d in disambig]

    # Join the list of roots back into a single string and return it
    return ' '.join(root)

def normalizeLetter(text:str) -> str:
    """
    Normalize Arabic letters by replacing certain letters with their more common equivalents as shown and return the modified text.
    
    ا <= أ
    ا <= إ
    ا <= آ
    ا <= ٱ
    ه <= ة
    ي <= ى
    ء <= ئ
    ء <= ؤ

    Args:
        text (str): A string of Arabic text to be normalized.

    Returns:
        str: The modified string with certain letters replaced.
    """
    # Define a translation table using the str.maketrans() method
    # The first argument is a string of characters to be replaced
    # The second argument is a string of replacement characters
    translation_table = str.maketrans('أ إ آ ٱ ى ة ؤ ئ', 'ا ا ا ا ي ه ء ء')
    
    # Apply the translation table to the input text using the translate() method
    return text.translate(translation_table)

def addingTokens(text:str) -> str:
    """
    Add special tokens to the input text to indicate the beginning and end of a sequence and return the modified text.
    This is important for certain NLP tasks that require the use of a separator between tokens
    
    Args:
        text (str): A string of text to which special tokens will be added.

    Returns:
        str: The modified string with special tokens added.
    """
    if text.startswith('[start]') and text.endswith('[end]'):
        # The tokens are already added, so return the input text as is
        return text
    else:
        # The tokens are not added, so add them and return the modified text
        return '[start] ' + text + ' [end]'
    
def removeExtraChar(text:str) -> str:
    
    # Handing the "اااووييي" expression 
    
    while "وو" in text:
        text = text.replace("وو", "و")

    # Double Spacing
    while "اا" in text:
        text = text.replace("اا", "ا")

    # Double Spacing
    while "يي" in text:
        text = text.replace("يي", "ي")

    # Double Spacing
    while "  " in text:
        text = text.replace("  ", " ")
    
    return text

def lemmatization(text:str) -> str:
    """
    Lemmatizes Arabic text using the Qalsadi lemmatizer.

    Args:
    - text: A string containing the Arabic text to be lemmatized.

    Returns:
    - A string containing the lemmas of the input text, separated by spaces.
    """

    # Import the Qalsadi lemmatizer module
    import qalsadi.lemmatizer

    # Create a new instance of the Lemmatizer class
    lemmer = qalsadi.lemmatizer.Lemmatizer()

    # Use the lemmatize_text method to lemmatize the input text
    lemmas = lemmer.lemmatize_text(text)

    # Join the lemmas into a single string separated by spaces
    return ' '.join(lemmas)

def textCleaning(text:str) -> str:
    """
    Clean the Arabic input text by performing several preprocessing steps.

    Args:
        text (str): A string of text to be cleaned.

    Returns:
        str: The cleaned string of text.
    """
    text = removePunctuation(text)
    text = removeTashkeel(text)
    # text = removeStopWords(text)
    # text = stemming(text)
    # text = lemmatization(text)
    text = normalizeLetter(text)
    # text = removeTashkeel(text)
    text = removeExtraChar(text)
    text = addingTokens(text)
    return text

def clean_Arabic_Dict(arabic_dict) -> dict:
    """
    Clean the values of an input dictionary that contains Arabic text by performing several text preprocessing steps.

    Args:
        input_dict (dict): A dictionary containing Arabic text to be cleaned.

    Returns:
        dict: A new dictionary with the cleaned values.
    """
    from tqdm.auto import tqdm

    # Create a new dictionary to store the normalized values
    clean_arabic_dict = {}

    # Loop through the keys and values of the input dictionary
    for key, value in tqdm(arabic_dict.items()):
        # Normalize each string in the list of values
        normalized_values = []
        for string in value:
            normalized_values.append(textCleaning(string))

        # Add the normalized values to the output dictionary
        clean_arabic_dict[int(key)] = normalized_values

    return clean_arabic_dict

def getIdToNumOfCation(dict:str) -> dict:
    """
    Returns a new dictionary that maps each ID in the input dictionary to the number of captions associated with it.

    Parameters:
    dict (dict): A dictionary where each key represents an ID and its corresponding value is a list of captions.

    Returns:
    dict: A new dictionary where each key is an ID from the input dictionary, and the corresponding value is an integer representing the number of captions associated with that ID.

    Example:
    >>> input_dict = {'id_1': ['caption_1', 'caption_2'], 'id_2': ['caption_3']}
    >>> getIdToNumOfCation(input_dict)
    {'id_1': 2, 'id_2': 1}
    """

    from collections import OrderedDict

    ID_TO_NUM_OF_CAPTION = OrderedDict()

    for key in dict.keys():
        ID_TO_NUM_OF_CAPTION[key] = len(dict[key])

    return ID_TO_NUM_OF_CAPTION

def DICT_TO_TEXT(dict:dict, path:str) -> None:
    """
    Writes the values of the input dictionary to a text file located at the specified path.

    Parameters:
    dict (dict): A dictionary where each key represents a group of values to be written to a file.
    path (str): The path to the file where the text will be written.

    Returns:
    None

    Example:
    >>> input_dict = {'group_1': ['value_1', 'value_2'], 'group_2': ['value_3']}
    >>> DICT_TO_TEXT(input_dict, 'output.txt')
    # The contents of output.txt will be:
    # value_1
    # value_2
    # value_3
    """
    with open(path, "w") as f:
        for value in dict.values():
            for item in value:
                if item.endswith("\n"):
                    item = item[:-1]  # remove the newline
                f.write(item + "\n")

def get_var_name(var:str) -> str:
    """
    Returns the name of a variable as a string.

    Parameters:
    var : A variable whose name is to be determined.

    Returns:
    str: A string representing the name of the input variable.

    Example:
    >>> x = 5
    >>> get_var_name(x)
    'x'
    """
    for name in globals():
        if globals()[name] is var:
            return name
    for name in locals():
        if locals()[name] is var:
            return name
    return None

from collections import OrderedDict
def split_and_save_dict(d: OrderedDict, n: int, filepath: str):
    """
    Splits an OrderedDict into n smaller OrderedDicts and saves them as separate JSON files in the specified filepath.

    Parameters:
        d (OrderedDict): The OrderedDict to be split.
        n (int): The number of splits to create.
        filepath (str): The path to the directory where the JSON files will be saved.

    Returns:
        None

    Example:
        >>> my_dict = OrderedDict([(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four'), (5, 'five')])
        >>> split_and_save_dict(my_dict, 3, 'output/')
        # Three JSON files will be created:
        # output/split_1.json: {'1': 'one', '2': 'two'}
        # output/split_2.json: {'3': 'three', '4': 'four'}
        # output/split_3.json: {'5': 'five'}
    """
    import numpy as np
    import json
    splits = np.array_split(list(d.items()), n)
    for i, split in enumerate(splits):
        split_dict = OrderedDict(split)
        with open(f"{filepath}/split_{i+1}.json", "w") as f:
            json.dump(split_dict, f)

def split_file(input_file, num_files, output_dir):
    """
    Splits a large text file into smaller files and saves them to a specified output directory.

    Parameters:
    input_file (str): The path of the input file to be split.
    num_files (int): The number of output files that the input file should be split into.
    output_dir (str): The path of the directory where the output files will be saved.

    Returns:
    None

    Example Usage:
    split_file("input_file_path", 3, "output_directory_path")
    """
    import numpy as np
    import os

    # Read the lines of the input file into a list
    with open(input_file+".txt", 'r') as f:
        lines = f.readlines()

    # Split the lines into N smaller arrays
    arrays = np.array_split(lines, num_files)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write each smaller array to a separate file
    for i, array in enumerate(arrays):
        # Open a new output file for each array and write its lines
        with open(f"{output_dir}/{input_file}_{i+1}.txt", 'w') as f:
            for j, line in enumerate(array):
                if j == len(array) - 1:
                    # Remove the newline character from the last line
                    line = line.rstrip('\n')
                if line.strip():  # Check if the line is not empty
                    f.write(line)

    # Print a message to indicate how many files were successfully split
    print(f"{num_files} files split successfully.")

def merge_files(input_folder_path, output_file_path):
    """
    Merges the contents of multiple files in a specified folder into a single output file.

    Parameters:
    input_folder_path (str): The path of the folder containing the input files.
    output_file_path (str): The path of the output file that will contain the merged contents.

    Returns:
    None

    Example Usage:
    merge_files("input_folder_path", "output_file_path")
    """
    import re
    import os

    # Get list of files in specified folder
    file_list = []
    for filename in os.listdir(input_folder_path):
        # If file is not a directory, add its path to the file list
        if os.path.isfile(os.path.join(input_folder_path, filename)):
            filename = input_folder_path + "/" + filename
            file_list.append(filename)

    # Sort the file list based on the numerical value found in the file names
    file_paths_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()))

    # Merge the contents of the files into a single output file
    with open(output_file_path, "w") as output_file:
        for file_path in file_paths_list:
            # Open each file in read mode and write its contents to the output file
            with open(file_path, "r") as input_file:
                output_file.write(input_file.read() + "\n")

    # Print a message to indicate how many files were successfully merged
    print(f"{len(os.listdir(input_folder_path))} files merged successfully")

def remove_first_line(file_path):
    """
    Removes the first line of a file located at the specified path.

    Parameters:
    file_path (str): The path to the file where the first line will be removed.

    Returns:
    None

    Example:
    >>> remove_first_line('example.txt')
    # The first line of example.txt will be removed.
    """
    # Open the file for reading
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read all lines from the file into a list
        lines = f.readlines()
    # Remove the first line from the list
    lines.pop(0)
    # Open the file for writing
    with open(file_path, 'w', encoding='utf-8') as f:
        # Write the remaining lines back to the file
        f.writelines(lines)

def TEXT_TO_DICT(path, reference):
    """
    Reads text from a file located at the specified path and creates a dictionary of captions with their corresponding IDs.

    Parameters:
    path (str): The path to the file containing the captions.
    reference (dict): A reference dictionary where each key represents an ID from the corpus, and the corresponding value is not used in this function.

    Returns:
    dict: An OrderedDict where each key represents an ID from the corpus, and the corresponding value is a list of captions.

    Example:
    >>> reference_dict = {'id_1': None, 'id_2': None, 'id_3': None, 'id_4': None, 'id_5': None}
    >>> TEXT_TO_DICT('captions.txt', reference_dict)
    # Assuming captions.txt contains 25 lines of text (5 captions for each of the 5 IDs in reference_dict), the function will output
    # OrderedDict([('id_1', ['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']),
    #              ('id_2', ['caption_6', 'caption_7', 'caption_8', 'caption_9', 'caption_10']),
    #              ('id_3', ['caption_11', 'caption_12', 'caption_13',cont'd:

    #                       'caption_14', 'caption_15']),
    #              ('id_4', ['caption_16', 'caption_17', 'caption_18', 'caption_19', 'caption_20']),
    #              ('id_5', ['caption_21', 'caption_22', 'caption_23', 'caption_24', 'caption_25'])])
    """
    ID_TO_CAPTION_AR = OrderedDict()
    ids = sorted(list(reference.keys()))
    with open(path, 'r') as file:
        lines = [line.rstrip('\n') for line in file.readlines()]
        batch_size = 5
        j = 0
        for i in range(0, len(lines), batch_size):
            ID_TO_CAPTION_AR[ids[j]] = lines[i:i+batch_size]
            j = j + 1
    return ID_TO_CAPTION_AR

def TRANSLATE_DICT(dict_to_translate):
    """
    Translates the captions in a dictionary from English to Arabic using the Google Translate API.

    Parameters:
    dict_to_translate (dict): A dictionary where each key represents an image ID and the corresponding value is a list of captions in English.

    Returns:
    dict: A dictionary where each key represents an image ID and the corresponding value is a list of captions in Arabic.

    Example:
    >>> my_dict = {'id_1': ['This is a cat', 'The cat is sleeping'], 'id_2': ['A dog is running', 'The dog is barking']}
    >>> TRANSLATE_DICT(my_dict)
    # The function will return a dictionary where the captions have been translated to Arabic:
    # {'id_1': ['هذه قطة', 'القط نائم'], 'id_2': ['الكلب يجري', 'الكلب ينبح']}
    """
    from googletrans import Translator
    from tqdm.auto import tqdm

    translator = Translator()
    
    ID_TO_CAPTION_AR = {}
    
    # key = image_id, value = [captions]
    for key, value in tqdm(dict_to_translate.items()):
        
        translated_value = []
        
        # translate each caption
        for sentence in value:
            translated_sentence = translator.translate(sentence, dest='ar').text
            translated_value.append(translated_sentence)
            
        ID_TO_CAPTION_AR[int(key)] = translated_value
        
    return ID_TO_CAPTION_AR

def PATH_TO_ID(annFile):
    """
    Creates a mapping between image IDs and their corresponding file paths using annotation data in a JSON file.

    Parameters:
    annFile (str): The path to the JSON file containing the annotation data.

    Returns:
    dict: A dictionary where each key is a file path and the corresponding value is the image ID.

    Example:
    >>> annotation_file = 'annotations.json'
    >>> PATH_TO_ID(annotation_file)
    # Assuming the JSON file contains data for three images with IDs 1, 2, and 3, and file names 'image1.jpg', 'image2.jpg', and 'image3.jpg', respectively:
    # {'image1.jpg': 1, 'image2.jpg': 2, 'image3.jpg': 3}
    """
    import json
    with open(annFile, 'r') as f:
        annotations = json.load(f)

    # Create the image ID to file path mapping
    PATH_TO_ID = {}
    for image in annotations['images']:
        file_name = image['file_name']
        PATH_TO_ID[image['id']] = file_name

    # Create the image path to ID mapping
    return {v: k for k, v in PATH_TO_ID.items()}

def helperFunction():
    from helper import push_file_to_github
    from helper import split_and_save_dict
    from helper import removePunctuation
    from helper import removeTashkeel
    from helper import removeStopWords
    from helper import stemming
    from helper import normalizeLetter
    from helper import addingTokens
    from helper import removeExtraChar
    from helper import lemmatization
    from helper import textCleaning
    from helper import clean_Arabic_Dict
    from helper import clean_English_Dict
    from helper import getIdToNumOfCation
    from helper import DICT_TO_TEXT
    from helper import get_var_name
    from helper import split_and_save_dict
    from helper import split_file
    from helper import merge_files
    from helper import remove_first_line
    from helper import TEXT_TO_DICT
    from helper import TRANSLATE_DICT
    from helper import PATH_TO_ID