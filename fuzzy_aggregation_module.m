%% Initialization
clear ; close all; clc

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



data = csvread('customDatasetTrain-40703_tuples-NORMALIZED.csv');
X=data(2:end,1:41);   %training set

data = csvread('Training_membership_to_clusters.csv');
training_membership=data(:,:);   %training_membership_to_clusters



pred_cluster1 = predict_output_on_cluster(cluster1_Theta1, cluster1_Theta2, X);
pred_cluster2 = predict_output_on_cluster(cluster2_Theta1, cluster2_Theta2, X);
pred_cluster3 = predict_output_on_cluster(cluster3_Theta1, cluster3_Theta2, X);
pred_cluster4 = predict_output_on_cluster(cluster4_Theta1, cluster4_Theta2, X);
pred_cluster5 = predict_output_on_cluster(cluster5_Theta1, cluster5_Theta2, X);
pred_cluster6 = predict_output_on_cluster(cluster6_Theta1, cluster6_Theta2, X);



m=size(X,1);
y_input=zeros(m,5);

for i=1:m
  pred_all_cluster_on_one_sample=[pred_cluster1(i,:)' pred_cluster2(i,:)' pred_cluster3(i,:)' pred_cluster4(i,:)' pred_cluster5(i,:)' pred_cluster6(i,:)'];
  y_input(i,:)=training_membership(i,:) * pred_all_cluster_on_one_sample';  
end

