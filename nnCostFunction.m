function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


a1=[ones(m,1) X];    %activation of input layer is input itself,adding a1_0=1

z2=Theta1 * a1';
a2=sigmoid(z2)' ;  
a2=[ones(m,1) a2]; %adding a2_0=1

z3=Theta2 * a2';
a3=sigmoid(z3);

%[max_element p]=max(a3);
%p=p';

hypothesis=a3;


Y=zeros(num_labels,m);   %10*5000

for i=1:m
  Y(y(i,1),i)=1;
endfor

t1=Theta1';
t2=Theta2';

J=(sum(sum(-(Y.*(log(hypothesis)))-((1-Y).*(log(1-hypothesis))))))/m + lambda* ( sum(sumsq(t1(2:end,:))) + sum(sumsq(t2(2:end,:))))/(2*m)



for i=1:m
  a1=[ones(1,1) X(i,:)];    %activation of input layer is input itself,adding a1_0=1
  z2=Theta1 * a1';
  a2=sigmoid(z2)';   
  a2=[ones(1,1) a2]; %adding a2_0=1
  z3=Theta2 * a2';
  a3=sigmoid(z3);
  
 % for layer 3(output layer)
  error_3=a3 - Y(:,i);
  
 % for layer 2(hidden layer)
% z2=[z2;ones(1,size(z2,2))];
 temp= Theta2' * error_3;
 error_2=temp(2:end,1).* sigmoidGradient(z2); 
 
% error_2 = error_2(2:end);
 
 delta_1=delta_1 + error_2 * a1;
 delta_2=delta_2 + error_3 * a2;
 
endfor


D_1 = zeros(size(Theta1));
D_2 = zeros(size(Theta2));



D_1(:,1)=delta_1(:,1) ./ m;
D_1(:,2:end)=( delta_1(:,2:end) + lambda * Theta1(:,2:end)  ) ./ m;
D_2(:,1)=delta_2(:,1) ./ m;
D_2(:,2:end)=( delta_2(:,2:end)+ lambda * Theta2(:,2:end)  ) ./ m;
Theta1_grad=D_1;
Theta2_grad=D_2 ;

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
