COPYRIGHT
=========

This package contains Matlab source code of Gaussian process approximation methods described in:

Paul Bodesheim and Alexander Freytag and Erik Rodner and Joachim Denzler:
"Approximations of Gaussian Process Uncertainties for Visual Recognition Problems".
Proceedings of the Scandinavian Conference on Image Analysis (SCIA), 2013.

Please cite that paper if you are using this code!

(LGPL) copyright by Paul Bodesheim and Alexander Freytag and Erik Rodner and Joachim Denzler


CONTENT
=======

learn_approxGP_model.m
test_approxGP_model.m
optimizeD.m   
README.txt
License.txt


USAGE
=====

- model learning: - Use the function "learn_approxGP_model" to learn a model: model = learn_approxGP_model(K, labels, mode, noise).
                  - Please refer to the documentation in learn_approxGP_model.m for explanations of input and output variables.

- model testing: - Use the function "test_approxGP_model" to test the model: scores = test_approxGP_model(model,Ks,Kss).
                 - Please refer to the documentation in test_approxGP_model.m for explanations of input and output variables.



NOTE
====

If you want to use the optimized approximation methods, it is advisable to have a function "minimize" for using gradients within the optimization.
Experimental results given in the paper have been computed using the function "minimize" from the GPML toolbox:

@MISC{Rasmussen10:GPML,
  author = {C. E. {Rasmussen} and H. {Nickisch}},
  title = {GPML Gaussian Processes for Machine Learning Toolbox},
  year = {2010},
  note = {\url{http://mloss.org/software/view/263/}},
}

