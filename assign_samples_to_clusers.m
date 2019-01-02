clear ; close all; clc

data=csvread("clusters_membership.csv");
[max cluster_index ]=max(data);


%save
csvwrite("cluster_index.csv", cluster_index');