% Testing approximate Gaussian process regression models according to the work:
%
% Paul Bodesheim and Alexander Freytag and Erik Rodner and Joachim Denzler:
% "Approximations of Gaussian Process Uncertainties for Visual Recognition Problems".
% Proceedings of the Scandinavian Conference on Image Analysis (SCIA), 2013.
%
% Please cite that paper if you are using this code!
%
% 
% function scores = test_approxGP_model(model,Ks,Kss)
%
% INPUT:
%   model -- approximation model obtained from the method learn_approxGP_model
%   Ks -- (N x M) matrix containing similarities/kernel values between N training and M test samples
%   Kss -- (M x 1) column vector containing self-similarities of M test samples
%
% OUTPUT: 
%   scores -- output scores (either approximate means or approximate variances) depending on model.mode (see learn_approxGP_model for details)
%
%
% (LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Joachim Denzler
%
function scores = test_approxGP_model(model,Ks,Kss)

    Kss=Kss+model.noise*ones(size(Kss)); % add noise term to self-similarities Kss

    if strcmp(model.mode,'GP-FA-Var') % compute fast approximate variances
       scores = Kss - (Ks.^2)'*model.invD;
    end

    if strcmp(model.mode,'GP-FA-Mean') % compute fast approximate means
       scores = Ks' * model.approxAlpha; 
    end

    if strcmp(model.mode,'GP-OA-Var') % compute optimized approximate variances
       scores = Kss - (Ks.^2)'*model.invD;
    end

    if strcmp(model.mode,'GP-OA-Mean') % compute optimized approximate means
       scores = Ks' * model.approxAlpha; 
    end
      
end
