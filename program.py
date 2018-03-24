import glob
from PIL import Image
import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint




from keras.layers.merge import concatenate
import nltk

def data_generator(batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        
        df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
        df = df.sample(frac=1)
        iter = df.iterrows()
        c = []
        imgs = []
        for i in range(df.shape[0]):
            x = next(iter)
            c.append(x[1][1])
            imgs.append(x[1][0])


        count = 0
        while True:
            for j, text in enumerate(c):
                current_image = encoding_train[imgs[j]]
                for i in range(len(text.split())-1):
                    count+=1
                    
                    partial = [word2idx[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    
                    # Initializing with zeros to create a one-hot encoding matrix
                    # This is what we have to predict
                    # Hence initializing it with vocab_size length
                    n = np.zeros(vocab_size)
                    # Setting the next word to 1 in the one-hot encoded matrix
                    n[word2idx[text.split()[i+1]]] = 1
                    next_words.append(n)
                    
                    images.append(current_image)

                    if count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
                        count = 0

def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x

def preprocess_input(x):
    x /= 255.
    #x -= 0.5
    #x *= 2.
    return x

def split_data(l):
    temp = []
    for i in img:
        if i[len(images):] in l:
            temp.append(i)
    return temp

token = 'Flickr8k_text/Flickr8k.token.txt'

captions = open(token, 'r').read().strip().split('\n')

d = {}
for i, row in enumerate(captions):
    row = row.split('\t')
    row[0] = row[0][:len(row[0])-2]
    if row[0] in d:
        d[row[0]].append(row[1])
    else:
        d[row[0]] = [row[1]]
        
images = 'Flicker8k_Dataset/'

img = glob.glob(images+'*.jpg')

train_images_file = 'Flickr8k_text/Flickr_8k.trainImages.txt'

train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

train_img = split_data(train_images)
len(train_img)

val_images_file = 'Flickr8k_text/Flickr_8k.devImages.txt'
val_images = set(open(val_images_file, 'r').read().strip().split('\n'))

val_img = split_data(val_images)

test_images_file = 'Flickr8k_text/Flickr_8k.testImages.txt'
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

test_img = split_data(test_images)

model = InceptionV3(weights='imagenet')

from keras.models import Model

new_input = model.input
hidden_layer = model.layers[-2].output

model_new = Model(new_input, hidden_layer)

try:
    encoding_train = joblib.load("encoding_train.pkl")
except Exception:
    
    encoding_train = {}
    for img in tqdm(train_img):
        encoding_train[img[len(images):]] = encode(img)
try:
    encoding_test = joblib.load("encoding_test.pkl")
except Exception:
    encoding_test = {}
    for img in tqdm(test_img):
        encoding_test[img[len(images):]] = encode(img)
    
train_d = {}
for i in train_img:
    if i[len(images):] in d:
        train_d[i] = d[i[len(images):]]
        
val_d = {}
for i in val_img:
    if i[len(images):] in d:
        val_d[i] = d[i[len(images):]]
        
test_d = {}
for i in test_img:
    if i[len(images):] in d:
        test_d[i] = d[i[len(images):]]
        
caps = []
for key, val in train_d.items():
    for i in val:
        caps.append('<start> ' + i + ' <end>')
        
words = [i.split() for i in caps]

try:
    unique = joblib.load("unique.pkl")
except Exception:
    unique = []
    for i in words:
        unique.extend(i)
    
unique = list(set(unique))

word2idx = {val:index for index, val in enumerate(unique)}

idx2word = {index:val for index, val in enumerate(unique)}

max_len = 0
for c in caps:
    c = c.split()
    if len(c) > max_len:
        max_len = len(c)
        
vocab_size = len(unique)

df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')

c = [i for i in df['captions']]

imgs = [i for i in df['image_id']]

samples_per_epoch = 0
for ca in caps:
    samples_per_epoch += len(ca.split())-1

embedding_size = 300

image_model = Sequential([
        Dense(embedding_size, input_shape=(2048,), activation='relu'),
        RepeatVector(max_len)
    ])
    
caption_model = Sequential([
        Embedding(vocab_size, embedding_size, input_length=max_len),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(300))
    ])

#https://stackoverflow.com/questions/46397258/how-to-merge-sequential-models-in-keras-2-0
final_model = Sequential([
        #Merge([image_model, caption_model], mode='concat', concat_axis=1),
        concatenate([image_model, caption_model], axis=1),
        Bidirectional(LSTM(256, return_sequences=False)),
        Dense(vocab_size),
        Activation('softmax')
    ])
    
final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

epochs = 8

for i in range(epochs):

    final_model.fit_generator(data_generator(128, encoding_train), samples_per_epoch=samples_per_epoch, epochs=1, 
                          verbose=1)
    final_model.save_weights("model" + str(i) + "h5")

