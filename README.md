# new_metric


## Prepare Conda Environment

Reinstall packages from the export file:

    conda create -n new_metric --file package-list.txt

## The main error function (Mean Adjusted Exponent Error)
$Mean AdjustedExponentr Error = \frac{1}{N}\sum_{i=1}^n|\hat{y}_i-y_i|^{exp}$

$exp = 2 - tanh(\frac{y_i-a}{b})\times(\frac{\hat{y}_i-y_i}{c})$

$a: center, b:critical range, c:slope$

