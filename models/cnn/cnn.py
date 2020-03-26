net['input'] = Input((input_shape,), dtype='int32', 
                    name='input')

net["embeddingCNN"] = Embedding(
                            output_dim=EMBEDDING_DIMENSION, 
                            input_dim=MAX_INDEX,
                            input_length=MAX_STRING_LENGTH, 
                            name='embeddingCNN')(net["input"])
# Parallel Conv Filters

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

# Global max pooling operation for each filter size

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

net["dropoutcnnmid"] = Dropout(0.5, 
                    name="dropoutcnnmid")(net["concatcnn"])

net["densecnn"] = Dense(NUM_CONV_FILTERS, activation="relu", 
                    name="densecnn")(net["dropoutcnnmid"])

net["dropoutcnn"] = Dropout(0.5, 
                    name="dropoutcnn")(net["densecnn"])

net["output"] = Dense(1, activation="sigmoid", 
                    name="densefinal")(net["dropoutcnn"])


model = Model(net['input'], net['output'])
