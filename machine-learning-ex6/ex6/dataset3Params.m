function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_t = [0.01,0.03,0.1, 0.3,1, 3,10,30];
sigma_t = [0.01,0.03,0.1, 0.3,1, 3,10,30];

error = 2000;


for n = 1:length(C_t)

  for m = 1:length(sigma_t)
  
    model = svmTrain(X, y, C_t(n), @(x1, x2) gaussianKernel(x1, x2, sigma_t(m)));
    predictions = svmPredict(model, Xval);
    error_t = mean(double(predictions ~= yval));
    if error_t < error
        error = error_t;
        C = C_t(n);
        sigma = sigma_t(m);  
        fprintf('\nPrediction error is %f with C %f and sigma %f', error, C_t(n), sigma_t(m));   
    end
    
  end;
    
end;



% =========================================================================

end
