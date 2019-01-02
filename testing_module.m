%% Initialization
clear ; close all; clc

cluster_centroid=csvread("cluster_centroid.csv");


data=csvread("customDatasetTest_18793_tuples.csv");
X=data(2:end,1:41);
y=data(2:end,42);


m=size(X,1);
k=6;  #no. of clusters
test_samples_membership=zeros(m,k);

for i=1:m
  for j=1:k
        res1=sumsq(X(i,:)-cluster_centroid(j,:));
        res3=0;
        for t=1:k
        res2=sumsq(X(i,:)-cluster_centroid(t,:));
        res3=res3+(res1/res2);
        endfor
    test_samples_membership(i,j)=1/(res3);
  endfor
endfor





