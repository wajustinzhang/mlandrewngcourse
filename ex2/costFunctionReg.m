function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


sum = 0;
regf = 0;

nSize = size(theta);
for i = 2: nSize
  regf = regf + theta(i)^2;
end 
regf = lambda * regf /(2*m);

for i = 1:m
  hi = sigmoid(theta' * X(i,:)');
  sum = sum + y(i) * log(hi) + (1-y(i))*log(1-hi) ;
  grad =  grad + (hi - y(i)) * X(i, :)';  
end

% theta1 should have no lambda adjust
grad(1) = (1/m) * grad(1);
grad(2:length(grad)) = (1/m) * grad(2:length(grad)) + (lambda/m) * theta(2:length(theta));
J = -(1/m) * sum + regf; 

% =============================================================

end
