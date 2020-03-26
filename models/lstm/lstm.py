model = Sequential()
model.add(Embedding(MAX_INDEX, EMBEDDING_DIMENSION, 
        input_length=MAX_STRING_LENGTH))
model.add(LSTM(256))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation("sigmoid"))
