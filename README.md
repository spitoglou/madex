# new_metric


## Setting up environment


### Anaconda

Anaconda is a free distribution of the Python programming language for large-scale data processing, predictive analytics, and scientific computing that aims to simplify package management and deployment.

Follow instructions to install [Anaconda](https://docs.continuum.io/anaconda/install) or the more lightweight [miniconda](http://conda.pydata.org/miniconda.html).

### Conda packages
In the root of the project there is a platform independent ```environment.yml``` file.

You can find up-to-date instructions on how to install and activate it [here (conda documentation)](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).


## The main error function (Mean Adjusted Exponent Error)
$Mean AdjustedExponentr Error = \frac{1}{N}\sum_{i=1}^n|\hat{y}_i-y_i|^{exp}$

$exp = 2 - tanh(\frac{y_i-a}{b})\times(\frac{\hat{y}_i-y_i}{c})$

$a: center, b:critical range, c:slope$

