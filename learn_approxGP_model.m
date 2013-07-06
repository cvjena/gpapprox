% Learning approximate Gaussian process regression models according to the work:
%
% Paul Bodesheim and Alexander Freytag and Erik Rodner and Joachim Denzler:
% "Approximations of Gaussian Process Uncertainties for Visual Recognition Problems".
% Proceedings of the Scandinavian Conference on Image Analysis (SCIA), 2013.
%
% Please cite that paper if you are using this code!
% 
%
% function model = learn_approxGP_model(K, labels, mode, noise)
%
% INPUT:
%   K -- (N x N) kernel matrix with N the number of training samples
%   labels -- (N x 1) column vector of labels for Gaussina process regression (each element equal to 1 for novelty detection, elements from {1,-1} for binary classification)
%   mode -- one of the following terms that indicates the approximation:
%              - GP-FA-Var (calculating fast approximate variances)
%              - GP-FA-Mean (calculating fast approximate means)
%              - GP-OA-Var (calculate optimized approximate variances)
%              - GP-OA-Mean (calculate optimized approximate means) 
%
%   noise -- assumed output noise, which will be added to the diagonal (variance of the Gaussian noise model)
%
% OUTPUT: 
%   model -- approximation model, which can be used as an argument for test_approxGP_model
%
%
% (LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Joachim Denzler
%
function model = learn_approxGP_model(K, labels, mode, noise)
   
    % store some information
    model.mode = mode;
    model.noise = noise;
    model.labels = labels;
    
    if strcmp(mode,'GP-FA-Var') % compute D^-1 for fast approximate variance computations
      model.invD = 1./sum( K + noise*eye(size(K)) , 2 );
    end

    if strcmp(mode,'GP-FA-Mean') % compute (D^-1)*labels for fast approximate mean computations
      model.approxAlpha = labels./sum( K + noise*eye(size(K)) , 2 );
    end

    if strcmp(mode,'GP-OA-Var') % compute D^-1 for optimized approximate variance computations
      invL = inv(chol( K + noise*eye(size(K)) )');
      invK = invL'*invL; % need the inverse of the regularized kernel matrix explicitely
      model.invD = -optimizeD(invK, -1./sum( K + noise*eye(size(K)) , 2 )); % use fast approximation as initialization
    end

    if strcmp(mode,'GP-OA-Mean') % compute (D^-1)*labels for optimized approximate mean computations
      invL = inv(chol( K + noise*eye(size(K)) )');
      invK = invL'*invL; % need the inverse of the regularized kernel matrix explicitely
      model.approxAlpha = (-optimizeD(invK, -1./sum( K + noise*eye(size(K)) , 2 ))).*labels; % use fast approximation as initialization
    end
    
end