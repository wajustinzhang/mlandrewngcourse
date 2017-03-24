function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(X); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
J = 0;

Diff = (X * theta - y);
J = 1/(2*m) * (sum(Diff.^2) + lambda*sum(theta(2:end).^2)); %theta0 doesn't need to caculate cost

% =========================================================================
grad(1,:) = (1/m) * sum(X(:,1)' * Diff );
for i=2:length(theta)
  grad(i,:) = (1/m) * sum(X(:,i)' * Diff ) + (lambda/m)* theta(i,:);
end;
  
grad = grad(:);

end
