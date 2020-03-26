net['input'] = Input((input_shape,), 
                    dtype='int32', 
                    name='input')

net["embedding"] = Embedding(
                        output_dim=EMBEDDING_DIMENSION, 
                        input_dim=MAX_INDEX,
                        input_length=MAX_STRING_LENGTH, 
                        name='embedding')(net["input"])

net['extradense'] = Dense(100, activation='relu', 
                        name="extradense")(net['embedding'])

net['flatten'] = Flatten()(net['extradense'])

net['dropout'] = Dropout(0.5, name="dropout")
                    (net['flatten'])

net['output'] = Dense(1, activation='sigmoid', 
                    name="output")(net['dropout'])

model = Model(net['input'], net['output'])
