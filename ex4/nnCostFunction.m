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

%Part 1.a

a1 = [ones(m, 1) X];
a2 = [ones(m, 1) sigmoid(a1 * Theta1')];
a3 = sigmoid(a2 * Theta2');

yVec = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
cost = -yVec .* log(a3) - (1 - yVec) .* log(1 - a3);

J = (1/m) * sum(sum(cost));

%Part 1.b
t1 = Theta1(:,2:end);
t2 = Theta2(:,2:end);
reg1 = sum(sum(t1.^2));
reg2 = sum(sum(t2.^2));

J = J + ( (lambda/(2*m)) * (reg1 + reg2));

%Part 2

D1 = zeros(size(Theta1)); % (25 x 401)
D2 = zeros(size(Theta2)); % (10 x 26)

for t = 1:m
    %step 1
    a1b = [1 X(t,:)]'; % (401 x 1)
    a2b = [1; sigmoid(Theta1 * a1b)]; % (26 x 1)
    a3b = sigmoid(Theta2 * a2b); % (10 x 1)
    %step 2
    yb = yVec(t,:); % (1 x 10)
    d3 = a3b - yb'; % (10 x 1)
    %step 3
    z2b = [1; Theta1 * a1b]; % (26 x 1)
    d2 = Theta2' * d3 .* sigmoidGradient(z2b); % (26 x 1)
    %step 4
    D1 = D1 + d2(2:end) * a1b'; % (25 x 401)
    D2 = D2 + d3 * a2b'; % (10 x 26)
end

Theta1_grad = (1/m) * D1;
Theta2_grad = (1/m) * D2;

%Part 3

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
