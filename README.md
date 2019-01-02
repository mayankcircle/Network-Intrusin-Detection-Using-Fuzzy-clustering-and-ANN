# Network Intrusion Detection Using Fuzzy Clustering And Artificial Neural Network
### Dataset-
 [NSL-KDD DATASET](http://www.unb.ca/cic/datasets/nsl.html) which is a multi class dataset having 41 features that is about the characterstics of a particular type of attack on the network.
Number Of Attacks/classes - 5 (DOS,USER TO REMOTE,PROBE, REMOTE TO LOCAL,UNKNOWN)
Number Of Features - 41
TRAINING SET - 5 million
TEST SET - 2 million


### Abstract-

> .According to many survey papers on Network Intrusion Detection, ANN performs better than traditional methods but here the problem is we have very less number of samples for attck type "probe", it leads to skewed distribution of classes(attacks) which leads to low precision. To overcome this situation, we use Fuzzy Clustering which partion the training dataset into many subsets, now we apply ANN to each subset, since subsets are less complex and of less size, each ANN can learn each subset more quickly, robustly and precisely, especially for low frequent attacks such as "user to remote". Now we use one big ANN as fuzzy Aggregation Module which combines the different ANNs results. This is same as "DIVIDE AND CONQUER".


### Installation

We need [Anaconda Distribution](https://www.anaconda.com/download/). Make sure that your python version should be >=3.5 otherwise create environment for that.
We need also [Octave](https://www.gnu.org/software/octave/download.html).



### Development

#### Preprocessing
For preprocessing, we use Python script named as " Data Processing On Research. py " where we first load NSL KDD DATASET named as "KDDTrain+.csv" and make Pandas Object for each type of attacks and then merge them all into one pandas object and randomize the dataset. By this way, we have all samples having valid attack type (not unknown).

Next, we change catagorical attributes to Numerical attributes and then we have to do standard normalize all attributes. This makes convergence of optimization function fast.

We save all preprocessed dataset in " customDatasetTrain-40703_tuples.csv " file.

#### Building the model
First we execute " main.m ", it does Fuzzy Clustering on training dataset and makes 6 clusters.  We stored the result of this in " cluster_membership.csv ".

Next, we execute " assign_samples_to_clusers.m ", here we assign each sample to particular cluster according to membership.We stored the result of this in " customDatasetTrain-40703_tuples-NORMALIZED with CLUSTER INDEX.csv ".

Next we call "ANN module.m" for each cluster, here we are fitting ANN with right regularized parameter. This ANN module has 41 input layer neurons, 18 hidden layer neurons and 5 output layer neurons (5 attack types-dos,r2l,u2r,probe,unknown). The optimal reduced cost function and corressponding theta are stored in txt file under folder named as "cluster".

Next, we need to aggregate the fuzzy ANNs, for this we use a big ANN. Here-
* We changed the feature set here, for each training sample (41 features), we computed prediction(cluser membership) to each clusters by previously built 6 ANN Modules. Now, for each sample we have 5 membership values, we multiplied this predictions matrix to original cluster membership matrix in fuzzy clustering result, which gives column matrix of 5 columns. By this way, we map 41 features to 5 features.

This ANN has 5 input layer neurons, 13 hidden layer neurons and 5 output layer neurons.

#### Result-
This model takes 3hr 15 min for training on i5 processor with 8GB RAM, After this, we get 89.33% accuracy on test set.We have to further tune this model, so that we can get along 94% accuracy. I thought that this will outperform the random forest model but it did not, Random forest model has 97% test set accuracy.
