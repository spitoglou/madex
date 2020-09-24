# new_metric

[![pipeline status](https://gitlab.csl.gr/spitoglou/new_metric/badges/master/pipeline.svg)](https://gitlab.csl.gr/spitoglou/new_metric/-/commits/master)
[![coverage report](https://gitlab.csl.gr/spitoglou/new_metric/badges/master/coverage.svg)](https://gitlab.csl.gr/spitoglou/new_metric/-/commits/master)

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


## Run Tests

Just run ```pytest``` while in the root directory.

### Coverage

To see coverage run ```pytest --cov```  

To create ```cov.xml``` file (in order to report coverage in the IDE) run
```shell
pytest --cov-report xml:cov.xml --cov
```


