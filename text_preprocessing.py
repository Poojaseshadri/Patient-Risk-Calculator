import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from symspellpy import SymSpell
from nltk.corpus import stopwords

# Download required NLTK data
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('words')

# Initialize SymSpell object
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = nltk.data.find('corpora/words').path + '/en'
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1, separator='\t')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define text preprocessing functions
def correct_spelling(text):
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    processed_text = ' '.join(tokens)
    processed_text = correct_spelling(processed_text)
    return processed_text

def preprocess_text_column(column):
    return column.apply(preprocess_text)

# Create a transformer for use in pipelines
text_preprocessor = FunctionTransformer(preprocess_text_column)