net['input'] = Input((input_shape,), dtype='int32', 
                    name='input')

########################
#          CNN         #
########################

net["embeddingCNN"] = Embedding(
                        output_dim=EMBEDDING_DIMENSION, 
                        input_dim=MAX_INDEX,
                        input_length=MAX_STRING_LENGTH, 
                        name='embeddingCNN')(net["input"])

# Parallel Convolutional Layer

net["conv2"] = Conv1D(NUM_CONV_FILTERS, 2, name="conv2")
                    (net["embeddingCNN"])

net["conv3"] = Conv1D(NUM_CONV_FILTERS, 3, name="conv3")
                    (net["embeddingCNN"])

net["conv4"] = Conv1D(NUM_CONV_FILTERS, 4, name="conv4")
                    (net["embeddingCNN"])

net["conv5"] = Conv1D(NUM_CONV_FILTERS, 5, name="conv5")
                    (net["embeddingCNN"])

net["conv6"] = Conv1D(NUM_CONV_FILTERS, 6, name="conv6")
                    (net["embeddingCNN"])

# Global max pooling

net["pool2"] = GlobalMaxPool1D(name="pool2")
                        (net["conv2"])

net["pool3"] = GlobalMaxPool1D(name="pool3")
                        (net["conv3"])

net["pool4"] = GlobalMaxPool1D(name="pool4")
                        (net["conv4"])

net["pool5"] = GlobalMaxPool1D(name="pool5")
                        (net["conv5"])

net["pool6"] = GlobalMaxPool1D(name="pool6")
                        (net["conv6"])


net["concatcnn"] = concatenate([net["pool2"],
                            net["pool3"], net["pool4"],
                            net["pool5"], net["pool6"]], 
                            axis=1, name='concatcnn')


net["dropoutcnnmid"] = Dropout(0.5, name="dropoutcnnmid")
                            (net["concatcnn"])

net["densecnn"] = Dense(NUM_CONV_FILTERS, activation="relu",
                        name="densecnn")(net["dropoutcnnmid"])

net["dropoutcnn"] = Dropout(0.5, name="dropoutcnn") 
                        (net["densecnn"])

########################
#         LSTM         #
########################

net["embeddingLSTM"] = Embedding(output_dim=max_features, 
                             input_dim=256,
                             input_length=MAX_STRING_LENGTH, 
                             name='embeddingLSTM')
                             (net["input"])


net["lstm"] = LSTM(256, name="lstm")(net["embeddingLSTM"])

net["dropoutlstm"] = Dropout(0.5, name="dropoutlstm")
                        (net["lstm"])

########################
#    Combine - ANN     #
########################

net['concat'] = concatenate([net['dropoutcnn'], 
                            net['dropoutlstm']], 
                            axis=-1, name='concat')

net['dropoutsemifinal'] = Dropout(0.5, name="dropoutsemifinal")
                            (net['concat'])

net['extradense'] = Dense(100, activation='relu', 
                        name="extradense")
                        (net['dropoutsemifinal'])

net['dropoutfinal'] = Dropout(0.5, name="dropoutfinal")
                        (net['extradense'])

net['output'] = Dense(1, activation='sigmoid', name="output")
                    (net['dropoutfinal'])

model = Model(net['input'], net['output'])
