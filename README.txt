INPUTS

ANOVA 1 Functions Inputs

	•	data_1: A list of numpy arrays, where each array represents a group and contains observations.
	•	C_1: Coefficient matrix to describe linear combinations of group means.
	•	d: The vector of constant terms for each linear combination.
	•	alpha: The significance level.

ANOVA 2 Functions Input

	•	X: Input data of shape (I, J, K). It should be a numpy array.
	•	alpha: The significance level.

Linear Regression Functions Inputs

	•	X_lr: The design matrix (n x (k + 1)), where n is the number of observations and k is the number of predictors (excluding the intercept).
	•	y_lr: The response vector (n x 1).
	•	C: The coefficient matrix (r x (k + 1)), where r is the rank of the coefficient matrix.
	•	c0: The vector of constants (r x 1) under the null hypothesis H0: Cβ = c0.
	•	indices: The indices of the coefficients to be tested for equality to zero.
	•	D: The matrix of predictors for which predictions are desired, of shape (m, k+1).
	•	alpha: The significance level.



FUNCTIONS

ANOVA Functions

	•	ANOVA1_partition_TSS: Calculate the partitioned sums of squares for a one-way ANOVA.
	•	ANOVA1_test_equality: Perform one-way ANOVA to test the equality of means among multiple groups.
	•	ANOVA1_is_contrast: Check if the sum of coefficients is zero, indicating a contrast.
	•	ANOVA1_is_orthogonal: Check if two sets of coefficients represent orthogonal contrasts.
	•	Bonferroni_correction: Calculate the Bonferroni-corrected significance level for multiple tests.
	•	Sidak_correction: Calculate the Šidák-corrected significance level for multiple tests.
	•	is_pairwise_contrast: Check if coefficients represent a pairwise contrast for Tukey’s HSD.
	•	Scheffe_test: Perform Scheffé’s test for multiple comparisons.
	•	Tukey_test: Perform Tukey’s HSD test for multiple comparisons.
	•	t_test: Perform t-tests for multiple comparisons.
	•	ANOVA1_CI_linear_combs: Calculate confidence intervals for linear combinations of means in a one-way ANOVA.
	•	ANOVA1_test_linear_combs: Perform an ANOVA test for linear combinations of means.
	•	ANOVA2_partition_TSS: Compute the partitioned sums of squares for a two-way ANOVA.
	•	ANOVA2_test_equality: Perform a two-way ANOVA test for equality of means.
	•	ANOVA2_MLE: Compute the maximum likelihood estimates (MLE) for a two-way ANOVA model.

Linear Regression Functions

	•	Mult_LR_Least_squares: Find the least squares solution for the multiple linear regression model.
	•	Mult_LR_partition_TSS: Compute the total sum of squares, regression sum of squares, and residual sum of squares.
	•	Mult_norm_LR_simul_CI: Compute simultaneous confidence intervals for the regression coefficients.
	•	Mult_norm_LR_CR: Compute the center, shape matrix, and radius of a multivariate normal confidence region.
	•	Mult_norm_LR_is_in_CR: Check if the coefficient vector is in the confidence region of a multivariate linear regression model.
	•	Mult_norm_LR_test_general: Perform a general linear hypothesis test for multiple linear regression.
	•	Mult_norm_LR_test_comp: Test the null hypothesis H0: β_j1 = … = β_jr = 0 in a multiple linear regression model.
	•	Mult_norm_LR_test_linear_reg: Test the existence of a linear regression relationship in multiple linear regression.
	•	Mult_norm_LR_pred_CI: Compute confidence intervals for multiple linear regression predictions using the multivariate normal assumption.


USAGE

	1.	Open main.py:
	•	Ensure you have the main.py file in your working directory.
	2.	Function inputs mentioned at the beginning of the README file are presented at the beginning of the main.py file:
	•	This includes the definitions for inputs such as alpha, data_1, C_1, d, X, X_lr, y_lr, D, etc.
	3.	Modify the inputs the way you want:
	•	Adjust the input values in main.py to suit your specific testing requirements. For example, you can change the values in data_1, the design matrix X_lr, the response vector y_lr, and any other parameters.
	4.	Run main.py:
	•	Execute the script by running python main.py in your terminal or IDE. This will call the functions from the toolbox and test them with the provided input data.
