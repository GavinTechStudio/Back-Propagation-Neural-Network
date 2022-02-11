$$
h_j = \sigma( \sum_i x_i w_{ij} - \beta_j )
$$

$$
\hat{y_k} = \sigma( \sum_j h_j v_{jk} - \lambda_k )
$$

$$
Loss = \frac{1}{2}\sum_k ( y_k - \hat{y_k} )^2
$$


$$
\Delta \lambda_k = - \eta (y_k - \hat{y_k}) \hat{y_k} (1 - \hat{y_k})
$$

$$
\Delta v_{jk} = \eta ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) h_j
$$

$$
\Delta \beta_j = - \eta \sum_k ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) v_{jk} h_j ( 1 - h_j )
$$

$$
\Delta w_{ij} = \eta \sum_k ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) v_{jk} h_j ( 1 - h_j ) x_i
$$





































