# Long Short-Term Memory (LSTM) Network

Deep learning first addressed DGA detection with work by [Woodbridge et al.](https://arxiv.org/abs/1611.00791), an implementation of an LSTM used for nonspecific DGA analysis. Their experiments show that their deep learning approach, an LSTM network, outperforms a character-level HMM and a random forest model that utilise features such as the entropy of character distribution. Their analysis and implementation led to a large success for identifying most DGA families; however, their LSTM did not score highly on `suppobox` or `matsnu`, dictionary DGA families. 

## A bit about LSTMs

Since we can treat domains as sequences of characters, LSTM  models are a natural fit for classifying DGA domains. LSTM nodes make decisions about one element in the sequence based on what it has seen earlier in the sequence. Thus, LSTM nodes learn parameters that are shared across the elements of sequence. This parameter sharing allows LSTMs to scale to handle much longer sequences than would be practical for traditional feedforward neural networks \[[Goodfellow, et. al](https://www.deeplearningbook.org/)\].

For example, an LSTM neuron might recall that it has seen seven vowels in a nine-character domain, making it unlikely that the domain is made up of natural English text. This sequential specialisation of LSTMs attracted us initially, but we found it alone could not generalise to new DGAs as well as other architectures.
