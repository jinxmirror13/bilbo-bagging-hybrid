# MIT Hybrid Model

Based on the original encoder-decoder model present by MIT \[[1](https://arxiv.org/abs/1607.07514)\],
several recent publications have adapted this CNN-LSTM hybrid model to DGA classification 
\[[2](http://faculty.washington.edu/mdecock/papers/byu2018a.pdf)\], 
\[[3](https://biblio.ugent.be/publication/8629567/file/8629568)\], 
\[[4](https://www.researchgate.net/publication/326855127_SPOOF_Net_Syntactic_Patterns_for_identification_of_Ominous_Online_Factors)\]. Unlike our model, this uses the CNN convolutions to feed inputs into an LSTM. The MIT hybrid architecture adapted by Yu et al. \[[2](http://faculty.washington.edu/mdecock/papers/byu2018a.pdf)\] is another benchmark during testing. Comparing Bilbo's parallel usage of a CNN and an LSTM to this model demonstrates the significance of our parallel architecture in binary classification of dictionary DGAs.

Their single convolutional layer consists of 128 one-dimensional filters, each three characters long with a stride of one. This is fed into a Max Pooling layer before a 64-node LSTM. This model contains no drop out and relies on a single sigmoid to flatten the results to a single score. 

## References

1. "Tweet2vec: Learning tweet embeddings using character-level cnn-lstm encoder-decoder" by Vosoughi, Soroush and Vijayaraghavan, Prashanth and Roy, Deb. Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval, pg. 1041--1044. 2016 ACM

2. "Character level based detection of DGA domain names" by Yu, Bin and Pan, Jie and Hu, Jiaming and Nascimento, Anderson and De Cock, Martine. 2018 International Joint Conference on Neural Networks (IJCNN), pg. 1--8. 2018 IEEE

3. "An evaluation of DGA classifiers" by Sivaguru, Raaghavi and Choudhary, Chhaya and Yu, Bin and Tymchenko, Vadym and Nascimento, Anderson and De Cock, Martine. 2018 IEEE International Conference on Big Data (Big Data), pg. 5058--5067. 2018 IEEE

4. "S.P.O.O.F Net: Syntactic Patterns for identification of Ominous Online Factors" by V. S. {Mohan} and V. {R} and S. {KP} and P. {Poornachandran}. 2018 IEEE Security and Privacy Workshops (SPW), pg. 258-263. DOI 10.1109/SPW.2018.00041. 2018 IEEE



