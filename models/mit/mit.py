
main_input = Input(shape=(MAX_STRING_LENGTH, ), 
                dtype='int32', name='main_input')

embedding = Embedding(input_dim=MAX_INDEX, 
                      output_dim=EMBEDDING_DIMENSION, 
                      input_length=MAX_STRING_LENGTH) 
                      (main_input)

conv = Conv1D(filters=128, kernel_size=3, padding='same', 
            activation='relu', strides=1) (embedding)

max_pool = MaxPool1D(pool_size=2, padding='same') (conv)

encode = LSTM(64, return_sequences=False) (max_pool)

output = Dense(1, activation='sigmoid') (encode)

model = Model(inputs=main_input, outputs=output)
