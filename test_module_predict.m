%% Initialization
clear ; close all; clc

test_samples_membership=csvread("test_samples_membership.csv");

[max cluster_index ]=max(test_samples_membership');





cluster1_Theta1=csvread("cluster 1 ANN OPTIMAL cost/Theta1.csv");
cluster1_Theta2=csvread("cluster 1 ANN OPTIMAL cost/Theta2.csv");
cluster2_Theta1=csvread("cluster 2 ANN OPTIMAL cost/Theta1.csv");
cluster2_Theta2=csvread("cluster 2 ANN OPTIMAL cost/Theta2.csv");
cluster3_Theta1=csvread("cluster 3 ANN OPTIMAL cost/Theta1.csv");
cluster3_Theta2=csvread("cluster 3 ANN OPTIMAL cost/Theta2.csv");
cluster4_Theta1=csvread("cluster 4 ANN OPTIMAL cost/Theta1.csv");
cluster4_Theta2=csvread("cluster 4 ANN OPTIMAL cost/Theta2.csv");
cluster5_Theta1=csvread("cluster 5 ANN OPTIMAL cost/Theta1.csv");
cluster5_Theta2=csvread("cluster 5 ANN OPTIMAL cost/Theta2.csv");
cluster6_Theta1=csvread("cluster 6 ANN OPTIMAL cost/Theta1.csv");
cluster6_Theta2=csvread("cluster 6 ANN OPTIMAL cost/Theta2.csv");



data = csvread('customDatasetTest_18793_tuples.csv');
X=data(2:end,1:41);   %test set
y=data(2:end,42);    %labels



fuzzy_aggregation_Theta1=csvread("fuzzy aggregation module cost/Theta1.csv");
fuzzy_aggregation_Theta2=csvread("fuzzy aggregation module cost/Theta2.csv");


m=size(X,1);


% CALL PARTICULAR ANN MODULE
for i=1:m
  if cluster_index(i)==1
    pred_cluster1 = predict_output_on_cluster(cluster1_Theta1, cluster1_Theta2, X(i,:));
    pred(i,1) = predict(fuzzy_aggregation_Theta1, fuzzy_aggregation_Theta2,pred_cluster1);
  endif
    if cluster_index(i)==2
   pred_cluster2 = predict_output_on_cluster(cluster2_Theta1, cluster2_Theta2, X(i,:));
   pred(i,1) = predict(fuzzy_aggregation_Theta1, fuzzy_aggregation_Theta2,pred_cluster2);
  endif
    if cluster_index(i)==3
   pred_cluster3 = predict_output_on_cluster(cluster3_Theta1, cluster3_Theta2,X(i,:));
   pred(i,1) = predict(fuzzy_aggregation_Theta1, fuzzy_aggregation_Theta2,pred_cluster3);
  endif
    if cluster_index(i)==4
    pred_cluster4 = predict_output_on_cluster(cluster4_Theta1, cluster4_Theta2, X(i,:));
    pred(i,1) = predict(fuzzy_aggregation_Theta1, fuzzy_aggregation_Theta2,pred_cluster4);
  endif
    if cluster_index(i)==5
    pred_cluster5 = predict_output_on_cluster(cluster5_Theta1, cluster5_Theta2,X(i,:));
    pred(i,1) = predict(fuzzy_aggregation_Theta1, fuzzy_aggregation_Theta2,pred_cluster5);
  endif
    if cluster_index(i)==6
   pred_cluster6 = predict_output_on_cluster(cluster6_Theta1, cluster6_Theta2,X(i,:));
   pred(i,1) = predict(fuzzy_aggregation_Theta1, fuzzy_aggregation_Theta2,pred_cluster6);
  endif
endfor

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y)) * 100);


