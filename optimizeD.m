% Optimization function to minimize the maximum eigenvalue of A(d) = A0 + Diag(d)
% with the additional constraint that the minimum eigenvalue is non-negative using 
% the smooth convex approximation of the maximum eigenvalue function from:
%
% [1] Xin Chen and Houduo Qi and Liqun Qi and Kok-Lay Teo:
% "Smooth Convex Approximation to the Maximum Eigenvalue Function"
% Journal of Global Optimization 30: 253--270, 2004
% 
% This function is used to learn approximate Gaussian process regression models according to the work:
%
% Paul Bodesheim and Alexander Freytag and Erik Rodner and Joachim Denzler:
% "Approximations of Gaussian Process Uncertainties for Visual Recognition Problems".
% Proceedings of the Scandinavian Conference on Image Analysis (SCIA), 2013.
%
% Please cite these papers if you are using this code!
%
% 
% function d = optimizeD(A0,d0)
%
% INPUT:
%   A0 -- (N x N) matrix to compute A(d) = A0 + Diag(d)
%   d0 -- (optional) (N x 1) column vector used for initialization
%
% OUTPUT: 
%   d -- (N x 1) column vector as the solution of the optimization
%
%
% (LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Joachim Denzler
%
function d = optimizeD(A0,d0)

  % check initialization
  if nargin < 2
    d0 = zeros(size(A0,1),1);
  end

  % some settings for the optimization
  EPS = 1.0;
  shrinkFactor = 0.8;
  maxDeltaNorm = 1e-16;
  options.MaxFunEvals = 10000;
  options.MaxIter = 10000;
  maxIterations = 100;
  
  % compute initial maximum eigenvalue
  f0 = max(eig(A0+diag(d0)));
  
  for i=1:maxIterations

    EPS = EPS * shrinkFactor;
    
    % use minimize method if available
    if exist('minimize') == 2
      % use a gradient-based approach via function "minimize"
      [d, f, numIterations] = minimize(d0, @(y) myfunepsgrad (y,A0,EPS), -5);
      f = f(end);
    else
      % apply simple Nelder-Mead method instead
      [d,f] = fminsearch( @(y) myfuneps(y,A0,EPS), d0, options);
    end
       
    % compute differences between solutions d and function values f
    deltaNorm = max(abs(d-d0));
    fDelta = abs(f-f0);
        
    if ~isfinite(f) % stop iterating if EPS became too small
      d = d0;
      disp(sprintf('optimizeD.m: stop at iteration %d of %d -- f = %f\n',i,maxIterations,f));
      break;
    elseif min(eig(A0+diag(d))) < 0 % check constraint that the minimum eigenvalue is >= 0
      disp(sprintf('optimizeD.m: resetting because of negative eigenvalue'));
      d = d0;
      continue;
    elseif deltaNorm < maxDeltaNorm % stop iterating if change in solutions is small
      disp(sprintf('low delta vector norm: %f\n', deltaNorm));
      break;
    else % prepare for next iteration
      d0 = d;
      f0 = f;
    end

  end
    
end % of function optimizeD

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [f, grad] = myfunepsgrad(y,A0,EPS)
% function [f, grad] = myfunepsgrad(y,A0,EPS)
%
% INPUT:
%   y -- (N x 1) column vector to compute A(y) = A0 + Diag(y)
%   A0 -- (N x N) matrix to compute A(y) = A0 + Diag(y)
%   EPS -- parameter of the optimization
%
% OUTPUT: 
%   f -- function value of the smooth convex approximation
%   grad -- corresponding gradient

  % check dimensions
  if size(A0,1) ~= size(A0,2) || size(A0,1) ~= length(y)
    error('myfunepsgrad: problem with the dimensions');
  end

  % get eigen-decomposition of A(y)=A0+Diag(y)
  [eigenvecs, eigenvals] = eig(A0+diag(y));
  eigenvals = diag(eigenvals);
  
  % size of the matrix
  n = length(y);
  
  % operator matrices
  A = zeros(n,n);
  
  % calculate A_tilde and the W matrix according to Eq. (13) and previous formulas of paper [1]
  W = zeros(n,n);
  for i=1:n
    A(i,i) = 1;
    A_tilde = eigenvecs' * A * eigenvecs;
    % Eq. (13) of paper [1]
    W(:,i) = diag(A_tilde);
    A(i,i) = 0;
  end
  
  % maximum eigenvalue
  eigmax = max(eigenvals);
  
  % calculate mu according to the first formula on p. 258 of paper [1]
  mu = exp((eigenvals - eigmax)./EPS);
  mu = mu ./ sum(mu);
  
  % regularization trade-off parameter
  %regul = norm(eig(A0+diag(y)));
  regul = 1;

  % calculate the gradient of the regularized stuff
  grad = W' * mu + regul*y;
    
  % compute value f
  f = eigmax + EPS*log(sum(exp((eigenvals-eigmax)./EPS))) + 0.5*regul*(y')*y;

end % of function myfunepsgrad

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function f = myfuneps(y,A0,EPS)
% function f = myfuneps(y,A0,EPS)
%
% INPUT:
%   y -- (N x 1) column vector to compute A(y) = A0 + Diag(y)
%   A0 -- (N x N) matrix to compute A(y) = A0 + Diag(y)
%   EPS -- parameter of the optimization
%
% OUTPUT: 
%   f -- function value of the smooth convex approximation
%

  % check dimensions
  if size(A0,1) ~= size(A0,2) || size(A0,1) ~= length(y)
    error('myfuneps: problem with the dimensions');
  end

  % obtain eigenvalues
  eigenvals = eig(A0+diag(y));

  % compute function value f using formulas on p. 264 of paper [1]
  f = EPS*log(sum(exp(eigenvals./EPS)))+0.5*y'*y;

end % of function myfuneps
