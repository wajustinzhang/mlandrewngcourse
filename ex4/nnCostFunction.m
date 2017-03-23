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

% add a first colum bias x0
X = [ones(m, 1) X];

% need to convert y into matrix of 5000*10
Y = zeros(m, num_labels); % 5000* 10
for i=1:m
  Y(i, y(i)) = 1;
end

% part 1, Feedforward the neural network and return the cost in the variable J
A = sigmoid(X * Theta1'); % hidden layer (5000 * 25)
% add A0
A = [ones(m,1) A]; % (2500, 26)
% output
H = sigmoid(A * Theta2'); %(5000, 10). This is the h(x)

% caculate J

% 1: caculate sum without regularization
mainP = 0;
for i=1:m
  mainP = mainP + Y(i,:)*log(H(i,:))' + (1-Y(i,:))*log(1-H(i,:))'; 
end

% 2 caculate regularization
sumt1 = 0;
for j=1:hidden_layer_size
  for k = 1:input_layer_size
    sumt1 = sumt1 + Theta1(j,k+1)^2;% it is implement to put k+1, not including the first one
  end
end

sumt2 = 0;
for j=1: num_labels
  for k = 1: hidden_layer_size
    sumt2 = sumt2 + Theta2(j,k+1)^2; % it is implement to put k+1, not including the first one
  end
end
reg = (lambda/(2*m)) * (sumt1 + sumt2);

% 3: get the J
J = -(1/m) * mainP  + reg;

% =========================================================================

D1 = 0; D2 = 0;
for i=1:m
  a1 = X(i,:)'; %401 x 1
  
  %hidden layer
  z2 = Theta1 * a1; %25x1
  a2 = [1;sigmoid(z2)]; %26x1
  
  z3 = Theta2 * a2; % 10 x 1
  a3 = sigmoid(z3); %this is hi, 10x1 
  
  delta3 = a3 - Y(i,:)'; % 10 x 1
  delta2 = (Theta2'(2:end, :) * delta3) .* sigmoidGradient(z2); % 25 *1
  
  D1 = D1 + delta2 * a1'; % 25 x 401
  D2 = D2 + delta3 * a2'; % 10*26 
end

Theta1_grad(:,1) = (1/m)*D1(:,1);
Theta1_grad(:,2:end) = (1/m)*D1(:,2:end) + (lambda/m) * Theta1(:,2:end);

Theta2_grad(:,1) = (1/m)*D2(:,1);
Theta2_grad(:,2:end) = (1/m)*D2(:,2:end) + (lambda/m) * Theta2(:,2:end);

%Theta2_grad = (1/m)*D2 + (lambda/m) * Theta2(:,2:end);
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
