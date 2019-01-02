clear ; close all; clc

data = csvread('customDatasetTrain-40703_tuples-NORMALIZED.csv');
X=data(2:end,1:41);



%FUZZY CLUSTERING
%-------------------------------------------
pkg load fuzzy-logic-toolkit;


[CC,SP,Ofn]=fcm(X,6);
%--------------------------------------------

