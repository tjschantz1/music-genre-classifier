# =============================================================================
# CPSC-57400, Spring II 2019
# Thomas J Schantz
# Project
# =============================================================================

# Import packages (some done in-line)
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import time
import myplotters # home-grown

# Define function for capturing run-time
print('\n=== Program Initiated ===')
start_time = time.time()
def timer(start,end):
   hours, rem = divmod(end-start, 3600)
   minutes, seconds = divmod(rem, 60)
   return('{:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))

# Import data
print('\n[>>> Importing data...]')
df0 = pd.read_csv('data/lyrics.csv')

# Remove sparse genres
keep_genres = ['Rock', 'Pop', 'Hip-Hop', 'Metal', 'Country']
df0 = df0[df0.genre.isin(keep_genres)]

# Remove duplicates
df0.drop_duplicates(subset=['artist', 'song'], keep='first', inplace=True)

# Remove unnecessary fields)
df0.drop(['index', 'song', 'year', 'artist'], axis=1, inplace=True)

# Remove unwanted genre & lyric entries
df0 = df0.dropna(axis=0, subset=['genre', 'lyrics'])
df0 = df0[(df0.genre != 'Other') & (df0.genre != 'Not Available')]

# Remove anything inside of [], (), or {}
df0['lyrics'] = df0['lyrics'].str.replace(r'[\(\[\{].*?[\)\]\}]', ' ')

# Remove anything before ":" or "-" (e.g. "Chorus:" or "Verse-")
df0['lyrics'] = df0['lyrics'].str.replace(r'\w+[:|-]', '') # WORKS

# Remove duplicate phrases (e.g. chorus's)
def remove_dupes(lyric):
    
    # Separate each line of lyric into new list
    line_list = [l.split(',') for l in ','.join(lyric).split('\n')]
    
    dupes_removed = []
    for line in line_list:
        if line not in dupes_removed:
            dupes_removed.append(line)
    dupes_removed = [entry+' ' for sublist in dupes_removed 
                     for entry in sublist] # flatten list of lines
    dupes_removed = ''.join((l) for l in dupes_removed)
    
    return dupes_removed

lyric_list = df0.lyrics.values.tolist()
dupes_removed = [remove_dupes([l]) for l in lyric_list]
df0['lyrics'] = dupes_removed

print('\n=== Data Import Complete ===')
print('--- Runtime =', timer(start_time, time.time()),'---')
new_time = time.time()
print('\n*** Number of Songs by Genre within Data ***')
print(df0.genre.value_counts())

# =============================================================================
# Data Preprocessing
# =============================================================================

# Define function for preprocessing data
def preprocess_lyrics(lyric_list):
    
    # Prepare variables/functions
    stop_words = set(stopwords.words('english'))
    wnl = nltk.WordNetLemmatizer()
    
    # Process lyric list
    word_list = [re.sub('[^\w\s]', '', lyric) for lyric in lyric_list] # remove punct
    token_list = [nltk.word_tokenize(lyric) for lyric in word_list] # tokenize words
    tokened_words = [entry for sublist in token_list 
                     for entry in sublist] # flatten list of tokens
    fdist = nltk.FreqDist(tokened_words)
    sparse_words = list(filter(lambda x: x[1]==1, fdist.items()))
    sparse_words = [i[0] for i in sparse_words] # words only
    undesired_words = stop_words|set(sparse_words)
    meaningful_tokens = [[w.lower() for w in tokens 
                          if not w.lower() in undesired_words] for tokens in token_list]
    
    # Lemmatize (i.e. remove pluralization & restore to root form)
    root_tokens = [[wnl.lemmatize(w, pos='v') for w in tokens] for tokens in meaningful_tokens]
    
    # Prepare output
    clean_lyrics = [' '.join(tokens) for tokens in root_tokens] # concat words to string
    token_count = [len(lyric) for lyric in token_list] # calc num of words in song
    tokened_words = [entry for sublist in meaningful_tokens 
                 for entry in sublist] # flatten list of final, prestemmed tokens
    
    return clean_lyrics, token_count, tokened_words

# Preprocess data
print('\n[>>> Cleaning data...]')
lyric_list = df0.lyrics.values.tolist()
clean_lyrics, token_count, tokened_words = preprocess_lyrics(lyric_list)
df0['clean_lyrics'], df0['wordCount'] = clean_lyrics, token_count

df0.to_csv('data/lyrics_cleaned.csv', index=False) # save to file for load later

print('\n=== Data Cleaning Complete ===')
print('--- Runtime =', timer(new_time, time.time()),'---')
new_time = time.time()

# Data visualization
print('\n[>>> Visualizing data...]\n')
df0LyricAgg = df0[['genre','wordCount']].groupby('genre').agg(['mean']).reset_index()
df0LyricAgg.columns = [''.join(x) for x in df0LyricAgg.columns.ravel()]
df0LyricAgg.wordCountmean = df0LyricAgg.wordCountmean.round()

myplotters.myBar(df0LyricAgg, "Song's Average Word Count by Genre", None, 
                 'genre', 'Song Genre', 'wordCountmean', None, 'Word Count Mean', 
                 ['Word Count Mean'], .75, None, None, None, None, 
                 list(df0LyricAgg.wordCountmean))

top_20 = [word for (word, count) in nltk.FreqDist(tokened_words).most_common(20)]
top_20c = [count for (word, count) in nltk.FreqDist(tokened_words).most_common(20)]
dfTop20 = pd.DataFrame({'Word':top_20, 'Count':top_20c})

myplotters.myBar(dfTop20, "Top 20 Most Common Words Found in Vocabulary", None, 
                 'Word', 'Word', 'Count', None, 'Word Count', 
                 ['Word Count'], .75, None, None, None, None, top_20c)

print('\n=== Data Visualization Complete ===')
print('--- Runtime =', timer(new_time, time.time()),'---')
new_time = time.time()

# =============================================================================
# Model Building
# =============================================================================

# Split train and test data
from sklearn.model_selection import train_test_split
lyrics = df0.clean_lyrics.values # train/test data, x
genres = df0.genre
train_lyrics, test_lyrics, train_y, test_y = train_test_split(
        lyrics, genres, test_size=0.3, random_state=42)
test_y_eval = np.copy(test_y) # create copy of test array for use in model eval

# Build embeddings
print('\n[>>> Building word embeddings and class encodings...]')
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras import utils

max_words = 5000 # vocab limit
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_lyrics) # word index lookup for vocab
train_x = tokenizer.texts_to_matrix(train_lyrics)
test_x = tokenizer.texts_to_matrix(test_lyrics)

# One-hot encode classes
encoder = LabelEncoder()
encoder.fit(train_y)
train_y = encoder.transform(train_y)
test_y = encoder.transform(test_y)
class_labels = list(np.unique((df0.genre)))
num_classes = len(class_labels) + 1 # 0 reserved for index
train_y = utils.to_categorical(train_y, num_classes)
test_y = utils.to_categorical(test_y, num_classes)

print('\n=== Word Embeddings & Class Encodings Complete ===')
print('--- Runtime =', timer(new_time, time.time()),'---')
new_time = time.time()

# Build neural network model
print('\n[>>> Building neural network model...]')

from keras.models import Sequential
from keras import layers

model = Sequential()
model.add(layers.Dense(512, input_shape=(max_words,), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

print('\n=== Neural Network Complete ===')
print('--- Runtime =', timer(new_time, time.time()),'---')
new_time = time.time()

# Train model
print('\n[>>> Training model...]\n')
mod_hist = model.fit(train_x, train_y, epochs=9, verbose=1, 
                     validation_data=(train_x, train_y),
                     batch_size=200)
loss, accuracy = model.evaluate(train_x, train_y, verbose=0)
print('\nTraining Accuracy: {:.2%}'.format(accuracy))
loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
print('Testing Accuracy: {:.2%}'.format(accuracy))
myplotters.plot_history(mod_hist)

print('\n=== Training Complete ===')
print('--- Runtime =', timer(new_time, time.time()),'---')
new_time = time.time()

# Serialize model to JSON for loading/evaluation later
model_json = model.to_json()
with open('genre_model.json', 'w') as json_file:
    json_file.write(model_json)

# Serialize weights to HDF5 for loading/evaluation later
model.save_weights('genre_model.h5')

# =============================================================================
# Model Evaluation
# =============================================================================

print('\n[>>> Performing model evaluation...]')

# Load pre-trained model
from keras.models import model_from_json
json_file = open('genre_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights into new model
loaded_model.load_weights('genre_model.h5')
 
# Evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', 
                     metrics=['accuracy'])
pred_y = loaded_model.predict_classes(test_x)

# Create dictionary of encoded:original labels
from more_itertools import unique_everseen
true_y = test_y_eval
original_label = list(unique_everseen(true_y)) # unique unordered
true_y = list(encoder.transform(true_y)) # encoded labels
encoded_label = list(unique_everseen(true_y)) # unique unordered
label_dict = dict(zip(encoded_label, original_label))

# Create confusion matrix
from sklearn.metrics import confusion_matrix
cm = pd.DataFrame(confusion_matrix(true_y, pred_y))
cm['col_total'] = cm.sum(axis=1)
cm.loc['row_total'] = cm.sum()

print('\n*** Genre Classification Performance Metrics ***\n')
for i in range(0, num_classes - 1):
    tp = cm.iloc[i,i]
    precision = tp / float(cm.iloc[i,5])
    recall = tp / float(cm.iloc[5,i])
    f1 = 2 * (precision*recall)/(precision+recall)
    print('{}:'.format(label_dict[i]))
    print('  >> precision = {:.2%}'.format(precision))
    print('  >> recall = {:.2%}'.format(recall))
    print('  >> F1-score = {:.2%}\n'.format(f1))

print('\n=== Evaluation Complete ===')
print('--- Runtime =', timer(new_time, time.time()),'---')
print('--- Total Runtime =', timer(start_time, time.time()),'---')