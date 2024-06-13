import numpy as np
import scipy.stats as stats
from scipy.stats import f, t, studentized_range
import itertools
import pandas as pd

def ANOVA1_partition_TSS(data: list, print_results=True):
    """
    Calculate the partitioned sums of squares for a one-way ANOVA.

    Parameters:
    data (list): A list of numpy arrays, where each array represents a group and contains observations.
    print_results (bool, optional): Whether to print the results. Defaults to True.

    Returns:
    tuple: A tuple containing the calculated sums of squares (SS_total, SS_within, SS_between).

    """
    # Flatten the list of arrays to calculate the grand mean
    all_data = np.concatenate(data)
    grand_mean = np.mean(all_data)
    
    # Initialize sums of squares
    SS_total = 0
    SS_within = 0
    SS_between = 0
    
    # Calculate SS_within and SS_between
    for group_data in data:
        group_mean = np.mean(group_data)
        n_i = len(group_data)
        
        # Compute within-group sum of squares
        SS_within += np.sum((group_data - group_mean) ** 2)
        
        # Compute between-group sum of squares
        SS_between += n_i * (group_mean - grand_mean) ** 2
    
    # Calculate SS_total
    SS_total = np.sum((all_data - grand_mean) ** 2)

    if print_results:
        print(f"SS_total: {SS_total}, SS_within: {SS_within}, SS_between: {SS_between}")

    return SS_total, SS_within, SS_between


def ANOVA1_test_equality(data: list, alpha=0.05):
    """
    Perform one-way ANOVA to test the equality of means among multiple groups.

    Parameters:
    - data: list of numpy arrays
        The input data containing multiple groups.
    - alpha: float, optional
        The significance level for the hypothesis test. Default is 0.05.

    Returns:
    None

    Prints the ANOVA table and the decision based on the p-value and alpha.

    """

    SS_total, SS_within, SS_between = ANOVA1_partition_TSS(data, print_results=False)
    
    # Calculate degrees of freedom
    df_between = len(data) - 1
    df_within = sum(len(group) for group in data) - len(data)
    df_total = df_between + df_within
    
    # Calculate mean squares
    MS_between = SS_between / df_between
    MS_within = SS_within / df_within
    
    # Calculate F statistic
    F_statistic = MS_between / MS_within
    
    # Calculate p-value
    p_value = f.sf(F_statistic, df_between, df_within)
    critical_value = f.ppf(1 - alpha, df_between, df_within)
    
    # Print the ANOVA table
    print(f"{'Source':<15} | {'df':>5} | {'SS':>10} | {'MS':>10} | {'F':>6}")
    print("-" * 55)
    print(f"{'Between groups':<15} | {df_between:5} | {SS_between:10.2f} | {MS_between:10.2f} | {F_statistic:6.2f}")
    print(f"{'Within groups':<15} | {df_within:5} | {SS_within:10.2f} | {MS_within:10.2f} | {'':6}")
    print(f"{'Total':<15} | {df_total:5} | {SS_total:10.2f} | {'':10} | {'':6}")
    
    # Decision
    if p_value < alpha:
        decision = 'Reject the null hypothesis'
    else:
        decision = 'Fail to reject the null hypothesis'
    
    print(f"\nP-value: {p_value:.4f}, Alpha: {alpha}")
    print(f"Critical value: {critical_value:.2f}, F-statistic: {F_statistic:.2f}")
    print("Decision:", decision)
    
    reject_null = p_value < alpha
    return critical_value, p_value, reject_null
    
def ANOVA1_is_contrast(coefficients):
    """
    Check if the sum of coefficients is zero, indicating whether the coefficients represent a contrast.

    Parameters:
    coefficients (list): A list of coefficients.

    Returns:
    bool: True if the sum of coefficients is zero, indicating a contrast. False otherwise.
    """
    is_contrast = [np.sum(row) == 0 for row in coefficients]
    dummy = 0
    return is_contrast

def ANOVA1_is_orthogonal(n, c1, c2, print_results=True):
    """
    Check if two sets of coefficients represent orthogonal contrasts.

    Parameters:
    n (list): A list of integers representing the number of observations in each group.
    c1 (list): A list of floats representing the coefficients of the first set of contrasts.
    c2 (list): A list of floats representing the coefficients of the second set of contrasts.

    Returns:
    bool: True if the contrasts are orthogonal, False otherwise.
    """

    # Check if both sets of coefficients are contrasts
    if sum(c1) != 0 or sum(c2) != 0:
        return "Warning: One or both sets of coefficients do not represent a contrast."
    
    num_groups = len(n)

    # Calculate the weighted sum for orthogonality
    weighted_sum = sum(c1[i] * c2[i] / n[i] for i in range(num_groups))
    
    # Determine if the contrasts are orthogonal
    is_orthogonal = abs(weighted_sum) < 1e-10  # use a small threshold to handle floating-point precision issues
    
    if print_results:
        if is_orthogonal:
            print("The linear combinations of means are orthogonal.")
        else:
            print("The linear combinations of means are not orthogonal.")
    return is_orthogonal

def Bonferroni_correction(alpha, m):
    """;
    Calculates the Bonferroni-corrected significance level for each of m tests.

    Parameters:
    alpha (float): The overall desired family-wise error rate (FWER).
    m (int): The number of tests being performed.

    Returns:
    float: The corrected significance level for each test.
    """
    corrected_alpha = alpha / m
    return corrected_alpha

def Sidak_correction(alpha, m):
    """
    Calculates the Šidák-corrected significance level for each of m tests.

    Parameters:
    alpha (float): The overall desired family-wise error rate (FWER).
    m (int): The number of tests being performed.

    Returns:
    float: The corrected significance level for each test.
    """
    corrected_alpha = 1 - (1 - alpha) ** (1 / m)
    return corrected_alpha

def is_pairwise_contrast(coefficients):
    
    # Pairwise contrasts checking for Tuckey's HSD
    
    non_zero_coeffs = [coeff for coeff in coefficients if coeff != 0]

    if len(non_zero_coeffs) == 2 and sum(coefficients) == 0 and abs(non_zero_coeffs[0]) == abs(non_zero_coeffs[1]):
        return True
    return False

def Scheffe_test(data, alpha, C, d):
    """
    Perform Scheffé's test for multiple comparisons.

    Parameters:
    data (list): A list of numpy arrays, where each array represents a group and contains observations.
    alpha (float): Significance level
    C (numpy array): Contrast matrix with shape (num_tests, num_groups)
    d (numpy array): Null hypothesis values for each contrast

    Returns:
    dict: Confidence intervals for each contrast
    """

    epsilon = 1e-10
    # Calculate basics
    grand_mean = np.mean(np.concatenate(data)) # X_bar_G
    group_means = [np.mean(group) for group in data] # X_bar_i
    num_groups = len(data) # I
    num_obs_per_group = [len(group) for group in data] # list of number of observations in each group
    num_obs_total = sum(num_obs_per_group) # total number of observations
    
    num_tests = C.shape[0]
    
    # Sums of squares
    SS_total, SS_within, SS_between = ANOVA1_partition_TSS(data, print_results=False)
    
    # Degrees of freedom
    df_total = num_obs_total - 1   # n-1
    df_between = num_groups - 1 # I-1
    df_within = df_total - df_between # n-I
    
    is_contrast = all(ANOVA1_is_contrast(C))
    
    CI = {}
    test_statistics = {}
    p_values = {}
    reject_nulls = {}

    for i in range(num_tests):
        if is_contrast:   # Theorem 2.8
            f_critical = f.ppf(1 - alpha, df_between, df_within)
            M_critical = np.sqrt(f_critical * df_between) # M = sqrt(F * (I-1))
        else:             # Theorem 2.7
            f_critical = f.ppf(1 - alpha, num_groups, df_within)
            M_critical = np.sqrt(f_critical * num_groups) # M = sqrt(F * I)
            
        contrast_mult_means = np.dot(C[i,:], group_means)  # C_j_i * X_bar_i
        contrast_variance = SS_within / df_within * np.sum(C[i,:] ** 2 / np.array(num_obs_per_group))   # MS_within * sum(C_j_i^2 / n_i)
        plus_minus = M_critical * np.sqrt(contrast_variance) # M * sqrt(MS_within * sum(C_j_i^2 / n_i))
        CI[i] = (contrast_mult_means - plus_minus, contrast_mult_means + plus_minus) # (C_j_i * X_bar_i - M * sqrt(MS_within * sum(C_j_i^2 / n_i)), C_j_i * X_bar_i + M * sqrt(MS_within * sum(C_j_i^2 / n_i)))
        
        test_statistics[i] = (contrast_mult_means - d[i]) / (np.sqrt(contrast_variance) + epsilon) # (C_j_i * X_bar_i - d_i) / sqrt(MS_within * sum(C_j_i^2 / n_i))
        p_values[i] = 1 - f.cdf(abs(test_statistics[i]) ** 1, df_between, df_within) # 1 - F(test_statistic^2)

        reject_nulls[i] = abs(test_statistics[i]) > M_critical
        
    return CI, p_values, reject_nulls

       

def Tukey_test(data, alpha, C, d):
    """
    Perform Tukey's HSD test for multiple comparisons.

    Parameters:
    data (list): A list of numpy arrays, where each array represents a group and contains observations.
    alpha (float): Significance level
    C (numpy array): Contrast matrix with shape (num_tests, num_groups)
    d (numpy array): Null hypothesis values for each contrast

    Returns:
    dict: Confidence intervals for each contrast
    """

    epsilon = 1e-10
    # Calculate basics
    grand_mean = np.mean(np.concatenate(data)) # X_bar_G
    group_means = [np.mean(group) for group in data] # X_bar_i
    num_groups = len(data) # I
    num_obs_per_group = [len(group) for group in data] # list of number of observations in each group
    num_obs_total = sum(num_obs_per_group) # total number of observations
    
    num_tests = C.shape[0]
    
    # Sums of squares
    SS_total, SS_within, SS_between = ANOVA1_partition_TSS(data, print_results=False)
    
    # Degrees of freedom
    df_total = num_obs_total - 1   # n-1
    df_between = num_groups - 1 # I-1
    df_within = df_total - df_between # n-I
    
    # Access each row of the contrast matrix and check if it is a pairwise contrast
    pairwise_check_list = []
    for row in range(len(C)):
        pairwise_check_list.append(is_pairwise_contrast(C[row]))
    is_pairwiseContrast = all(pairwise_check_list)
    
    CI ={}
    test_statistics = {}
    p_values = {}
    reject_nulls = {}
    
    if is_pairwiseContrast:
        for i in range(num_tests):
            contrast_mult_means = np.dot(C[i,:], group_means)
            contrast_variance = SS_within / df_within * np.sum(C[i,:] ** 2 / np.array(num_obs_per_group))
            studentized_critical = studentized_range.ppf(1 - alpha, num_groups, df_within) # studentized critical value
            plus_minus = studentized_critical / np.sqrt(2) * np.sqrt(contrast_variance)
            CI[i] = (contrast_mult_means - plus_minus, contrast_mult_means + plus_minus)
            
            test_statistics[i] = (contrast_mult_means - d[i]) / (np.sqrt(contrast_variance) + epsilon)
            p_values[i] = 1 - studentized_range.cdf(abs(test_statistics[i]) * np.sqrt(2), num_groups, df_within)

            reject_nulls[i] = abs(test_statistics[i]) > studentized_critical / np.sqrt(2)
            
        return CI, p_values, reject_nulls
    


def t_test(data, alpha, C, d):
    """
    Perform t-tests for multiple comparisons.

    Parameters:
    data (list): A list of numpy arrays, where each array represents a group and contains observations.
    alpha (float): Significance level
    C (numpy array): Contrast matrix with shape (num_tests, num_groups)
    d (numpy array): Null hypothesis values for each contrast

    Returns:
    dict: Confidence intervals for each contrast
    """

    epsilon = 1e-10
    # Calculate basics
    grand_mean = np.mean(np.concatenate(data)) # X_bar_G
    group_means = [np.mean(group) for group in data] # X_bar_i
    num_groups = len(data) # I
    num_obs_per_group = [len(group) for group in data] # list of number of observations in each group
    num_obs_total = sum(num_obs_per_group) # total number of observations
    
    num_tests = C.shape[0]
    
    # Sums of squares
    SS_total, SS_within, SS_between = ANOVA1_partition_TSS(data, print_results=False)
    
    # Degrees of freedom
    df_total = num_obs_total - 1   # n-1
    df_between = num_groups - 1 # I-1
    df_within = df_total - df_between # n-I
    
    CI = {}
    test_statistics = {}
    p_values = {}
    reject_nulls = {}
    
    for i in range(num_tests):
        contrast_mult_means = np.dot(C[i,:], group_means)
        t_critical = t.ppf(1 - alpha / 2, df_within)
        coefficient_variance = SS_within / df_within * np.sum(C[i,:] ** 2 / np.array(num_obs_per_group))
        plus_minus = t_critical * np.sqrt(coefficient_variance)
        CI[i] = (contrast_mult_means - plus_minus, contrast_mult_means + plus_minus)
        
        test_statistics[i] = (contrast_mult_means - d[i]) / (np.sqrt(coefficient_variance) + epsilon)
        p_values[i] = (1 - t.cdf(abs(test_statistics[i]), df_within)) * 2
        reject_nulls[i] = abs(test_statistics[i]) > t_critical
        
    return CI, p_values, reject_nulls


def ANOVA1_CI_linear_combs(data, C, alpha, method='Scheffe'):
    """
    Calculate confidence intervals for linear combinations of means in a one-way ANOVA.

    Parameters:
    - data: list of numpy arrays, representing the data for the ANOVA. Each element corresponds to a group, and each array contains observations.
    - alpha: float, the significance level for the confidence intervals.
    - C: numpy array, representing the contrast matrix for the linear combinations of means.
    - method: str, optional, the method to use for calculating the confidence intervals. Default is 'Scheffe'.

    Returns:
    - CI: dict, a dictionary containing the confidence intervals for each linear combination of means.

    Raises:
    - ValueError: if an unknown method is provided.
    """    

    # Calculate basics
    grand_mean = np.mean(np.concatenate(data)) # X_bar_G
    group_means = [np.mean(group) for group in data] # X_bar_i
    num_groups = len(data) # I
    num_obs_per_group = [len(group) for group in data] # list of number of observations in each group
    num_obs_total = sum(num_obs_per_group) # total number of observations
    
    num_tests = C.shape[0]
    
    d = np.zeros(num_tests)
    
    # Contrasts and Pairwise checking
    is_contrast = all(ANOVA1_is_contrast(C))
    
    pairwise_check_list = []
    for row in range(len(C)):
        pairwise_check_list.append(is_pairwise_contrast(C[row]))
    is_pairwise = all(pairwise_check_list)
    
    # Find orthogonality relations between each pair of contrasts
    all_combinations = list(itertools.combinations(list(C), 2))
    orthogonality_list = []
    for combo in all_combinations:
        orthogonality_list.append(ANOVA1_is_orthogonal(num_obs_per_group, combo[0], combo[1], print_results=False))
    is_orthogonal = all(orthogonality_list)
    
    CI = {}
    
    if method == 'Scheffe':
        CI, _, _ = Scheffe_test(data, alpha, C, d)
        if is_contrast:
            print("It is a contrast, so theorem 2.8 is used")
        else:
            print("It is not a contrast, so theorem 2.7 is used")
        return CI
        
    elif method == 'Tukey':
        if is_pairwise:
            CI, _, _ = Tukey_test(data, alpha, C, d)
            return CI
        else:
            print("Tukey's HSD can only be used for pairwise contrasts.")
            return None
    
    elif method == 'Bonferroni':
        corrected_alpha = Bonferroni_correction(alpha, num_tests) 
        CI, _, _ = t_test(data, corrected_alpha, C, d)
        return CI
    
    elif method == 'Sidak':
        if is_orthogonal:
            corrected_alpha = Sidak_correction(alpha, num_tests)
            CI, _, _ = t_test(data, corrected_alpha, C, d)
            return CI
        else:
            print("Sidak's method can only be used for orthogonal contrasts.")
            return None
        
    elif method == 'Best':
        if is_contrast:
            if is_orthogonal and not is_pairwise: # orthogonal and not pairwise
                print("It is a contrast and orthogonal but not pairwise")
                print("Comparison between Sidak and Scheffe(2.8)")
                corrected_alpha = Sidak_correction(alpha, num_tests)
                sidak_CI, _, _ = t_test(data, corrected_alpha, C, d)                
                scheffe_CI, _, _ = Scheffe_test(data, alpha, C, d)
                
                sidak_plus_minus = sum(sidak_CI[i][1] - sidak_CI[i][0] for i in range(num_tests))
                scheffe_plus_minus = sum(scheffe_CI[i][1] - scheffe_CI[i][0] for i in range(num_tests))
                                
                if sidak_plus_minus < scheffe_plus_minus:
                    print("Sidak CI is better")
                    return sidak_CI
                else:
                    print("Scheffe CI is better")
                    return scheffe_CI
                
            elif not is_orthogonal and not is_pairwise: # not orthogonal and not pairwise
                print("It is contrast but not pairwise and not orthogonal")
                print("Comparison between Bonferroni and Scheffe(2.8)")
                corrected_alpha = Bonferroni_correction(alpha, num_tests)
                bonferroni_CI, _, _ = t_test(data, corrected_alpha, C, d)
                scheffe_CI, _, _ = Scheffe_test(data, alpha, C, d)
                
                bonferroni_plus_minus = sum(bonferroni_CI[i][1] - bonferroni_CI[i][0] for i in range(num_tests))
                scheffe_plus_minus = sum(scheffe_CI[i][1] - scheffe_CI[i][0] for i in range(num_tests))
                
                if bonferroni_plus_minus < scheffe_plus_minus:
                    print("Bonferroni CI is better")
                    return bonferroni_CI
                else:
                    print("Scheffe CI is better")
                    return scheffe_CI
                
            elif not is_orthogonal and is_pairwise: # not orthogonal and pairwise
                print("It is a pairwise contrast and not orthogonal")
                print("Comparison between Bonferroni and Tukey")
                corrected_alpha = Bonferroni_correction(alpha, num_tests)
                bonferroni_CI, _, _ = t_test(data, corrected_alpha, C, d)
                tukey_CI, _, _ = Tukey_test(data, alpha, C, d)
                
                bonferroni_plus_minus = sum(bonferroni_CI[i][1] - bonferroni_CI[i][0] for i in range(num_tests))
                tukey_plus_minus = sum(tukey_CI[i][1] - tukey_CI[i][0] for i in range(num_tests))
                
                if bonferroni_plus_minus < tukey_plus_minus:
                    print("Bonferroni CI is better")
                    return bonferroni_CI
                else:
                    print("Tukey CI is better")
                    return tukey_CI
                
            elif is_orthogonal and is_pairwise: # orthogonal and pairwise
                print("It is a pairwise contrast and orthogonal")
                print("Comparison between Sidak and Tukey")
                corrected_alpha = Sidak_correction(alpha, num_tests)
                sidak_CI, _, _ = t_test(data, corrected_alpha, C, d)
                tukey_CI, _, _ = Tukey_test(data, alpha, C, d)
                
                sidak_plus_minus = sum(sidak_CI[i][1] - sidak_CI[i][0] for i in range(num_tests))
                tukey_plus_minus = sum(tukey_CI[i][1] - tukey_CI[i][0] for i in range(num_tests))
                                
                if sidak_plus_minus < tukey_plus_minus:
                    print("Sidak CI is better")
                    return sidak_CI
                else:
                    print("Tukey CI is better")
                    return tukey_CI
        else:
            print('Coefficients do not represent a contrast.')
            print("Comparison between Bonferroni and Scheffe(2.7)")
            corrected_alpha = Bonferroni_correction(alpha, num_tests)
            bonferroni_CI, _, _ = t_test(data, corrected_alpha, C, d)
            scheffe_CI, _, _ = Scheffe_test(data, alpha, C, d)
            
            bonferroni_plus_minus = sum(bonferroni_CI[i][1] - bonferroni_CI[i][0] for i in range(num_tests))
            scheffe_plus_minus = sum(scheffe_CI[i][1] - scheffe_CI[i][0] for i in range(num_tests))
                        
            if bonferroni_plus_minus < scheffe_plus_minus:
                print("Bonferroni CI is better")
                return bonferroni_CI
            else:
                print("Scheffe CI is better")
                return scheffe_CI
    else:
        raise ValueError(f"Unknown method: {method}")


def ANOVA1_test_linear_combs(data, C, alpha, d=None, method='Scheffe'):
    """
    Perform an ANOVA test for linear combinations of means.

    Parameters:
    - data (list): The data as a list of numpy arrays, where each array represents a group and contains observations.
    - C (ndarray): The contrast matrix with shape (m, I), where m is the number of linear combinations.
    - alpha (float): The significance level for hypothesis testing.
    - d (ndarray, optional): The vector of constant terms for each linear combination. Default is None.
    - method (str, optional): The method to use for multiple comparisons. Default is 'Scheffe'.

    Returns:
    - p_values (ndarray): The p-values for each hypothesis test.
    - reject_nulls (dict): A dictionary indicating whether each null hypothesis is rejected or not.

    Raises:
    - ValueError: If the dimensions of the input matrices are not valid.
    """
    
    # Calculate basics
    grand_mean = np.mean(np.concatenate(data))  # X_bar_G
    group_means = [np.mean(group) for group in data]  # X_bar_i
    num_groups = len(data)  # I
    num_obs_per_group = [len(group) for group in data]  # n_i
    num_obs_total = sum(num_obs_per_group)  # total number of observations
    
    num_tests = C.shape[0]
    
    if d is None:
        d = np.zeros(num_tests)
    
    # Contrasts and Pairwise checking
    is_contrast = all(ANOVA1_is_contrast(C))
    
    pairwise_check_list = []
    for row in range(len(C)):
        pairwise_check_list.append(is_pairwise_contrast(C[row]))
    is_pairwise = all(pairwise_check_list)
    
    # Find orthogonality relations between each pair of contrasts
    all_combinations = list(itertools.combinations(list(C), 2))
    orthogonality_list = []
    for combo in all_combinations:
        orthogonality_list.append(ANOVA1_is_orthogonal(num_obs_per_group, combo[0], combo[1], print_results=False))
    is_orthogonal = all(orthogonality_list)
    
    if method == 'Scheffe':
        CI, p_values, reject_nulls = Scheffe_test(data, alpha, C, d)
        if is_contrast:
            print("It is a contrast, so theorem 2.8 is used")
        else:
            print("It is not a contrast, so theorem 2.7 is used")

        if any(list(reject_nulls.values())):
            print("At least one null hypothesis is rejected.")
        else:
            print("No null hypothesis is rejected.")
        return p_values, reject_nulls
         
    elif method == 'Tukey':
        if is_pairwise:
            CI, p_values, reject_nulls = Tukey_test(data, alpha, C, d)

            if any(list(reject_nulls.values())):
                print("At least one null hypothesis is rejected.")
            else:
                print("No null hypothesis is rejected.")
            return p_values, reject_nulls
        
        else:
            print("Tukey's HSD can only be used for pairwise contrasts.")
            return None, None
    
    elif method == 'Bonferroni':
        corrected_alpha = Bonferroni_correction(alpha, num_tests) 
        CI, p_values, reject_nulls = t_test(data, corrected_alpha, C, d)

        if any(list(reject_nulls.values())):
            print("At least one null hypothesis is rejected.")
        else:
            print("No null hypothesis is rejected.")
        return p_values, reject_nulls
    
    elif method == 'Sidak':
        if is_orthogonal:
            corrected_alpha = Sidak_correction(alpha, num_tests)
            CI, p_values, reject_nulls = t_test(data, corrected_alpha, C, d)

            if any(list(reject_nulls.values())):
                print("At least one null hypothesis is rejected.")
            else:
                print("No null hypothesis is rejected.")
            return p_values, reject_nulls
        
        else:
            print("Sidak's method can only be used for orthogonal contrasts.")
            return None, None
        
    elif method == 'Best':
        if is_contrast:
            if is_orthogonal and not is_pairwise: # orthogonal and not pairwise
                print("It is a contrast and orthogonal but not pairwise")
                print("Comparison between Sidak and Scheffe(2.8)")
                corrected_alpha = Sidak_correction(alpha, num_tests)
                sidak_CI, sidak_p_values, sidak_reject_nulls = t_test(data, corrected_alpha, C, d)                
                scheffe_CI, scheffe_p_values, scheffe_reject_nulls = Scheffe_test(data, alpha, C, d)
                
                sidak_plus_minus = sum(sidak_CI[i][1] - sidak_CI[i][0] for i in range(num_tests))
                scheffe_plus_minus = sum(scheffe_CI[i][1] - scheffe_CI[i][0] for i in range(num_tests))
                                
                if sidak_plus_minus < scheffe_plus_minus:
                    print("Sidak is better")
                    p_values = sidak_p_values
                    reject_nulls = sidak_reject_nulls
                else:
                    print("Scheffe is better")
                    p_values = scheffe_p_values
                    reject_nulls = scheffe_reject_nulls
                
                if any(list(reject_nulls.values())):
                    print("At least one null hypothesis is rejected.")
                else:
                    print("No null hypothesis is rejected.")
                return p_values, reject_nulls
                
            elif not is_orthogonal and not is_pairwise:
                print("It is contrast but not pairwise and not orthogonal")
                print("Comparison between Bonferroni and Scheffe(2.8)")
                corrected_alpha = Bonferroni_correction(alpha, num_tests)
                bonferroni_CI, bonferroni_p_values, bonferroni_reject_nulls = t_test(data, corrected_alpha, C, d)
                scheffe_CI, scheffe_p_values, scheffe_reject_nulls = Scheffe_test(data, alpha, C, d)
                
                bonferroni_plus_minus = sum(bonferroni_CI[i][1] - bonferroni_CI[i][0] for i in range(num_tests))
                scheffe_plus_minus = sum(scheffe_CI[i][1] - scheffe_CI[i][0] for i in range(num_tests))
                
                if bonferroni_plus_minus < scheffe_plus_minus:
                    print("Bonferroni is better")
                    p_values = bonferroni_p_values
                    reject_nulls = bonferroni_reject_nulls
                else:
                    print("Scheffe is better")
                    p_values = scheffe_p_values
                    reject_nulls = scheffe_reject_nulls
                
                if any(list(reject_nulls.values())):
                    print("At least one null hypothesis is rejected.")
                else:
                    print("No null hypothesis is rejected.")
                return p_values, reject_nulls
                
            elif not is_orthogonal and is_pairwise:
                print("It is a pairwise contrast and not orthogonal")
                print("Comparison between Bonferroni and Tukey")
                corrected_alpha = Bonferroni_correction(alpha, num_tests)
                bonferroni_CI, bonferroni_p_values, bonferroni_reject_nulls = t_test(data, corrected_alpha, C, d)
                tukey_CI, tukey_p_values, tukey_reject_nulls = Tukey_test(data, alpha, C, d)
                
                bonferroni_plus_minus = sum(bonferroni_CI[i][1] - bonferroni_CI[i][0] for i in range(num_tests))
                tukey_plus_minus = sum(tukey_CI[i][1] - tukey_CI[i][0] for i in range(num_tests))
                
                if bonferroni_plus_minus < tukey_plus_minus:
                    print("Bonferroni is better")
                    p_values = bonferroni_p_values
                    reject_nulls = bonferroni_reject_nulls
                else:
                    print("Tukey is better")
                    p_values = tukey_p_values
                    reject_nulls = tukey_reject_nulls
                    
                if any(list(reject_nulls.values())):
                    print("At least one null hypothesis is rejected.")
                else:
                    print("No null hypothesis is rejected.")
                return p_values, reject_nulls                
            
            elif is_orthogonal and is_pairwise:
                print("It is a pairwise contrast and orthogonal")
                print("Comparison between Sidak and Tukey")
                corrected_alpha = Sidak_correction(alpha, num_tests)
                sidak_CI, sidak_p_values, sidak_reject_nulls = t_test(data, corrected_alpha, C, d)
                tukey_CI, tukey_p_values, tukey_reject_nulls = Tukey_test(data, alpha, C, d)
                
                sidak_plus_minus = sum(sidak_CI[i][1] - sidak_CI[i][0] for i in range(num_tests))
                tukey_plus_minus = sum(tukey_CI[i][1] - tukey_CI[i][0] for i in range(num_tests))
                                
                if sidak_plus_minus < tukey_plus_minus:
                    print("Sidak is better")
                    p_values = sidak_p_values
                    reject_nulls = sidak_reject_nulls
                else:
                    print("Tukey is better")
                    p_values = tukey_p_values
                    reject_nulls = tukey_reject_nulls
                
                if any(list(reject_nulls.values())):
                    print("At least one null hypothesis is rejected.")
                else:
                    print("No null hypothesis is rejected.")
                return p_values, reject_nulls
            
        else:
            print('Coefficients do not represent a contrast.')
            print("Comparison between Bonferroni and Scheffe(2.7)")
            corrected_alpha = Bonferroni_correction(alpha, num_tests)
            bonferroni_CI, bonferroni_p_values, bonferroni_reject_nulls = t_test(data, corrected_alpha, C, d)
            scheffe_CI, scheffe_p_values, scheffe_reject_nulls = Scheffe_test(data, alpha, C, d)
        
            bonferroni_plus_minus = sum(bonferroni_CI[i][1] - bonferroni_CI[i][0] for i in range(num_tests))
            scheffe_plus_minus = sum(scheffe_CI[i][1] - scheffe_CI[i][0] for i in range(num_tests))
                        
            if bonferroni_plus_minus < scheffe_plus_minus:
                print("Bonferroni is better")
                p_values = bonferroni_p_values
                reject_nulls = bonferroni_reject_nulls
            else:
                print("Scheffe is better")
                p_values = scheffe_p_values
                reject_nulls = scheffe_reject_nulls    
                        
            if any(list(reject_nulls.values())):
                print("At least one null hypothesis is rejected.")
            else:
                print("No null hypothesis is rejected.")
                
            return p_values, reject_nulls
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    
def ANOVA2_partition_TSS(X):
    """
    Compute the partitioned sums of squares for a two-way ANOVA.

    Parameters:
    X (ndarray): Input data array of shape (I, J, K), where I is the number of levels for factor A,
            J is the number of levels for factor B, and K is the number of replicates.

    Returns:
    tuple: A tuple containing the partitioned sums of squares:
        - SS_Total: Total sum of squares
        - SS_A: Sum of squares for factor A
        - SS_B: Sum of squares for factor B
        - SS_AB: Sum of squares for the interaction between factors A and B
        - SS_E: Sum of squares for the error

    """
    # Convert input X to a numpy array if it's not already one
    X = np.array(X)
    
    # Compute overall mean
    grand_mean = np.mean(X)
    
    # Number of levels for factors A and B, and number of replicates
    I, J, K = X.shape
    
    # Compute SS_Total
    SS_Total = np.sum((X - grand_mean) ** 2)
    
    # Compute main effects
    mean_A = np.mean(X, axis=(1, 2))  # Mean over J and K
    mean_B = np.mean(X, axis=(0, 2))  # Mean over I and K
    SS_A = J * K * np.sum((mean_A - grand_mean) ** 2)
    SS_B = I * K * np.sum((mean_B - grand_mean) ** 2)
    
    # Compute interaction effect SS_AB
    mean_AB = np.mean(X, axis=2)  # Mean over K only
    SS_AB = 0
    for i in range(I):
        for j in range(J):
            SS_AB += K * (mean_AB[i, j] - mean_A[i] - mean_B[j] + grand_mean)**2
    
    # Compute SS_E
    SS_E = SS_Total - (SS_A + SS_B + SS_AB)

    return SS_Total, SS_A, SS_B, SS_AB, SS_E

def ANOVA2_MLE(X):
    """
    Compute the maximum likelihood estimates (MLE) for a two-way ANOVA model.

    Parameters:
    X (ndarray): Input data of shape (I, J, K).

    Returns:
    tuple: A tuple containing the following MLE estimates:
        - mu (float): Overall mean of the data.
        - a_i (ndarray): Main effect of factor A.
        - b_j (ndarray): Main effect of factor B.
        - sigma_ij (ndarray): Interaction effect between factors A and B.
    """
    X = np.array(X)
    I, J, K = X.shape

    # Compute overall mean (mu)
    mu = np.mean(X)

    # Compute main effects
    mean_A = np.mean(X, axis=(1, 2))  # Mean over J and K
    mean_B = np.mean(X, axis=(0, 2))  # Mean over I and K
    a_i = mean_A - mu
    b_j = mean_B - mu

    # Adjust a_i and b_j to ensure they sum to zero
    a_i -= np.mean(a_i)
    b_j -= np.mean(b_j)

    # Compute interaction effects
    mean_AB = np.mean(X, axis=2)  # Mean over K only
    sigma_ij = mean_AB - mu - np.tile(a_i, (J, 1)).T - np.tile(b_j, (I, 1))

    # Adjust sigma_ij to ensure it sums to zero
    sigma_ij -= np.mean(sigma_ij, axis=0, keepdims=True)
    sigma_ij -= np.mean(sigma_ij, axis=1, keepdims=True)
    sigma_ij += np.mean(sigma_ij)

    return mu, a_i, b_j, sigma_ij

def ANOVA2_test_equality(X, alpha, test):
    """
    Perform a two-way ANOVA test for equality of means.

    Parameters:
    X (ndarray): The input data array of shape (I, J, K).
    alpha (float): The significance level.
    test (str): The type of test to perform. Can be "A", "B", or "AB".

    Returns:
    None

    Prints the results of the ANOVA test for the specified source.

    """
    X = np.array(X)
    I, J, K = X.shape

    # Compute overall mean (mu)
    mu = np.mean(X)
    
    # Compute main effects
    mean_A = np.mean(X, axis=(1, 2))
    mean_B = np.mean(X, axis=(0, 2))
    
    SS_Total, SS_A, SS_B, SS_AB, SS_E = ANOVA2_partition_TSS(X)
    
    # Degrees of freedom
    df_A = I - 1
    df_B = J - 1
    df_AB = (I - 1) * (J - 1)
    df_E = I * J * (K - 1)
    df_Total = I * J * K - 1
    
    # Mean squares
    MS_A = SS_A / df_A
    MS_B = SS_B / df_B
    MS_AB = SS_AB / df_AB
    MS_E = SS_E / df_E
    
    # F-statistics
    F_A = MS_A / MS_E
    F_B = MS_B / MS_E
    F_AB = MS_AB / MS_E
    
    # Output relevant test
    if test == "A":
        p_value = 1 - stats.f.cdf(F_A, df_A, df_E)
        result = "Significant" if p_value < alpha else "Not significant"
        print(f"Source: A, df: {df_A}, SS: {SS_A}, MS: {MS_A}, F: {F_A}, p-value: {p_value}, Result: {result}")
    
    elif test == "B":
        p_value = 1 - stats.f.cdf(F_B, df_B, df_E)
        result = "Significant" if p_value < alpha else "Not significant"
        print(f"Source: B, df: {df_B}, SS: {SS_B}, MS: {MS_B}, F: {F_B}, p-value: {p_value}, Result: {result}")
    
    elif test == "AB":
        p_value = 1 - stats.f.cdf(F_AB, df_AB, df_E)
        result = "Significant" if p_value < alpha else "Not significant"
        print(f"Source: AB, df: {df_AB}, SS: {SS_AB}, MS: {MS_AB}, F: {F_AB}, p-value: {p_value}, Result: {result}")
    
    return p_value, result


def Mult_LR_Least_squares(X, y):
    """
    Finds the least squares solution for the multiple linear regression model.
    
    Parameters:
    X (np.ndarray): The design matrix.
    y (np.ndarray): The response vector.
    
    Returns:
    beta_hat (np.ndarray): Maximum likelihood estimates for beta.
    sigma2_hat_ML (float): Maximum likelihood estimate for sigma^2.
    sigma2_hat_unbiased (float): Unbiased estimate for sigma^2.
    """
    # Number of observations and predictors
    n, p = X.shape
    
    # Compute beta_hat (MLE)
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Compute residuals
    residuals = y - X @ beta_hat
    
    # Compute sigma2_hat_ML (MLE for sigma^2)
    sigma2_hat_ML = (residuals.T @ residuals) / n
    
    # Compute sigma2_hat_unbiased (Unbiased estimate for sigma^2)
    sigma2_hat_unbiased = (residuals.T @ residuals) / (n - p)
    
    return beta_hat, sigma2_hat_ML, sigma2_hat_unbiased

def Mult_LR_partition_TSS(X, y):
    """
    Computes the total sum of squares, regression sum of squares, and residual sum of squares.
    
    Parameters:
    X (np.ndarray): The design matrix (n x (k + 1)).
    y (np.ndarray): The response vector (n x 1).
    
    Returns:
    TSS (float): Total sum of squares.
    RegSS (float): Regression sum of squares.
    RSS (float): Residual sum of squares.
    """
    # Number of observations
    n = X.shape[0]
    
    # Compute the mean of y
    y_mean = np.mean(y)
    
    # Compute beta_hat (MLE for beta)
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Compute the predicted values
    y_pred = X @ beta_hat
    
    # Compute the residuals
    residuals = y - y_pred
    
    # Compute TSS (Total Sum of Squares)
    TSS = np.sum((y - y_mean) ** 2)
    
    # Compute RSS (Regression Sum of Squares)
    RegSS = np.sum((y_pred - y_mean) ** 2)
    
    # Compute ESS (Error Sum of Squares or Residual Sum of Squares)
    RSS = np.sum(residuals ** 2)
    
    print(f"TSS: {TSS}, RegSS: {RegSS}, RSS: {RSS}")
    
    return TSS, RegSS, RSS

def Mult_norm_LR_simul_CI(X, y, alpha=0.05):
    """
    Computes simultaneous confidence intervals for the regression coefficients.
    
    Parameters:
    X (np.ndarray): The design matrix (n x (k + 1)).
    y (np.ndarray): The response vector (n x 1).
    alpha (float): Significance level.
    
    Returns:
    ci (np.ndarray): Confidence intervals for the regression coefficients.
    """
    # Number of observations and predictors
    n, k_plus_1 = X.shape
    
    # Compute beta_hat (MLE for beta)
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # Compute residuals
    residuals = y - X @ beta_hat
    
    # Compute the unbiased estimate for sigma^2
    Se_2 = (residuals.T @ residuals) / (n - k_plus_1)
    Se = np.sqrt(Se_2)
    
    # Compute the standard errors of the beta estimates
    se_beta_hat = Se * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    
    bonferroni_critical_value = t.ppf(1 - alpha / (2 * k_plus_1), df=n - k_plus_1)
    
    scheffe_critical_value = np.sqrt(k_plus_1 * f.ppf(1 - alpha, dfn=k_plus_1, dfd=n - k_plus_1))

    if bonferroni_critical_value < scheffe_critical_value:
        critical_value = bonferroni_critical_value
    else:
        critical_value = scheffe_critical_value    
    
    # Compute the confidence intervals
    ci = np.zeros((k_plus_1, 2))
    for i in range(k_plus_1):
        ci[i, 0] = beta_hat[i] - critical_value * se_beta_hat[i]
        ci[i, 1] = beta_hat[i] + critical_value * se_beta_hat[i]
    
    return ci


def Mult_norm_LR_CR(X, y, C, alpha=0.05):
    """
    Compute the center, shape matrix, and radius of a multivariate normal confidence region.

    Parameters:
    X (numpy.ndarray): The design matrix with shape (n, k_plus_1), where n is the number of observations and k_plus_1 is the number of predictors plus one.
    y (numpy.ndarray): The response vector with shape (n,).
    C (numpy.ndarray): The contrast matrix with shape (r, k_plus_1), where r is the number of rows in the contrast matrix.
    alpha (float, optional): The significance level for the confidence region. Defaults to 0.05.

    Returns:
    tuple: A tuple containing the center of the ellipsoid (C @ beta_hat), the shape matrix (inverse of C @ XTX_inv @ C.T), and the radius of the confidence region.

    """
    
    # Number of observations and predictors
    n, k_plus_1 = X.shape
    r = np.linalg.matrix_rank(C)
    
    XTX_inv = np.linalg.inv(X.T @ X)

    # Compute beta_hat (MLE for beta)
    beta_hat = XTX_inv @ X.T @ y
    
    # Compute the unbiased estimate for sigma^2
    residuals = y - X @ beta_hat
    Se_2 = (residuals.T @ residuals) / (n - k_plus_1)
    
    # Compute the center of the ellipsoid (C @ beta_hat)
    center = C @ beta_hat
    
    df1 = r
    df2 = X.shape[0] - X.shape[1]

    # Compute the critical value using the F-distribution
    f_critical = f.ppf(1 - alpha, df1, df2)
    
    radius_squared = r * Se_2 * f_critical
    radius = np.sqrt(radius_squared)
    
    shape_matrix = C @ XTX_inv @ C.T

    return center, shape_matrix, radius


def Mult_norm_LR_is_in_CR(X, y, C, c0, alpha):
    """
    Checks if the coefficient vector 'c0' is in the confidence region (CR) of a multivariate linear regression model.
    
    Parameters:
    X (numpy.ndarray): The design matrix of shape (n, k_plus_1), where 'n' is the number of observations and 'k_plus_1' is the number of predictors plus one.
    y (numpy.ndarray): The response vector of shape (n,).
    C (numpy.ndarray): The contrast matrix of shape (r, k_plus_1), where 'r' is the number of linearly independent contrasts.
    c0 (numpy.ndarray): The coefficient vector to be tested of shape (r,).
    alpha (float): The significance level for the hypothesis test.
    
    Returns:
    bool: True if the coefficient vector 'c0' is in the confidence region (CR), False otherwise.
    """
    # Number of observations and predictors
    n, k_plus_1 = X.shape
    r = np.linalg.matrix_rank(C)
    
    XTX_inv = np.linalg.inv(X.T @ X)

    # Compute beta_hat (MLE for beta)
    beta_hat = XTX_inv @ X.T @ y
    
    # Compute the unbiased estimate for sigma^2
    residuals = y - X @ beta_hat
    Se_2 = (residuals.T @ residuals) / (n - k_plus_1)
    
    # Compute the center of the ellipsoid (C @ beta_hat)
    center = C @ beta_hat
    
    df1 = r
    df2 = X.shape[0] - X.shape[1]

    # Compute the critical value using the F-distribution
    f_critical = f.ppf(1 - alpha, df1, df2)
    
    radius_squared = r * Se_2 * f_critical
    radius = np.sqrt(radius_squared)
    
    shape_matrix = np.linalg.inv(C @ XTX_inv @ C.T)
    
    test_statistics = (C @ beta_hat - c0).T @ shape_matrix @ (C @ beta_hat - c0) / Se_2 / r
    
    if test_statistics <= f_critical:
        print("co is in CR")
        return True

    else: 
        print("co is not in CR")
        return False


    
def Mult_norm_LR_test_general(X, y, C, c0, alpha):
    """
    Performs a general linear hypothesis test for multiple linear regression.

    Parameters:
    ----------
    X : ndarray
        The design matrix (n x (k + 1)), where n is the number of observations
        and k is the number of predictors (excluding the intercept).
    y : ndarray
        The response vector (n x 1).
    C : ndarray
        The contrast matrix (r x (k + 1)), where r is the rank of the contrast matrix.
    c0 : ndarray
        The vector of constants (r x 1) under the null hypothesis H0: Cβ = c0.
    alpha : float
        The significance level for the test.

    Returns:
    -------
    reject_H0 : bool
        True if the null hypothesis H0: Cβ = c0 is rejected, False otherwise.
    """

    # Number of observations and predictors
    n, k_plus_1 = X.shape
    r = np.linalg.matrix_rank(C)
    
    XTX_inv = np.linalg.inv(X.T @ X)

    # Compute beta_hat (MLE for beta)
    beta_hat = XTX_inv @ X.T @ y
    
    # Compute the unbiased estimate for sigma^2
    residuals = y - X @ beta_hat
    Se_2 = (residuals.T @ residuals) / (n - k_plus_1)
    
    # Compute the center of the ellipsoid (C @ beta_hat)
    center = C @ beta_hat
    
    df1 = r
    df2 = X.shape[0] - X.shape[1]

    # Compute the critical value using the F-distribution
    f_critical = f.ppf(1 - alpha, df1, df2)
    
    radius_squared = r * Se_2 * f_critical
    radius = np.sqrt(radius_squared)
    
    shape_matrix = np.linalg.inv(C @ XTX_inv @ C.T)
    
    test_statistics = (C @ beta_hat - c0).T @ shape_matrix @ (C @ beta_hat - c0) / Se_2 / r
    reject_null = test_statistics > f_critical
    
    if reject_null:
        print("Reject the null hypothesis")
        return reject_null
    else:
        print("Fail to reject the null hypothesis")
        return reject_null


def Mult_norm_LR_test_comp(X, y, indices, alpha):
    """
    Tests the null hypothesis H0: β_j1 = ... = β_jr = 0 in a multiple linear regression model.

    Parameters:
    ----------
    X : ndarray
        The design matrix (n x (k + 1)), where n is the number of observations
        and k is the number of predictors (excluding the intercept).
    y : ndarray
        The response vector (n x 1).
    alpha : float
        The significance level for the test.
    indices : list of int
        The indices of the coefficients to be tested for equality to zero.

    Returns:
    -------
    reject_H0 : bool
    True if the null hypothesis H0: β_j1 = ... = β_jr = 0 is rejected, False otherwise.
    """

    # Number of parameters (including intercept)
    n , k_plus_1 = X.shape
    
    # Construct C matrix
    C = np.zeros((len(indices), k_plus_1))
    for idx, j in enumerate(indices):
        C[idx, j] = 1
    
    # Construct c0 vector
    c0 = np.zeros(len(indices))
    
    # Perform the hypothesis test using the general function
    return Mult_norm_LR_test_general(X, y, C, c0, alpha)


def Mult_norm_LR_test_linear_reg(X, y, alpha):
    """
    Tests the existence of a linear regression relationship in multiple linear regression.

    Parameters:
    ----------
    X : ndarray
        The design matrix (n x (k + 1)), where n is the number of observations
        and k is the number of predictors (excluding the intercept).
    y : ndarray
        The response vector (n x 1).
    alpha : float
        The significance level for the test.

    Returns:
    -------
    reject_H0 : bool
        True if the null hypothesis H0: β_1 = ... = β_k = 0 is rejected, False otherwise.
    test_statistic : float
        The value of the test statistic.
    f_critical : float
        The critical value from the F-distribution.

    Notes:
    -----
    This function tests the null hypothesis H0: β_1 = ... = β_k = 0 versus the alternative
    hypothesis H1: not H0. It uses the F-distribution to determine the critical value
    and performs the test at the specified significance level α.
    """

    # Number of parameters (including intercept)
    _, k_plus_1 = X.shape
    
    # Indices for testing all coefficients (excluding the intercept) equal to zero
    indices = list(range(1, k_plus_1))
    
    # Perform the hypothesis test
    return Mult_norm_LR_test_comp(X, y, indices, alpha)

def Mult_norm_LR_pred_CI(X, y, D, alpha, method="Best"):
    """
    Compute confidence intervals for multiple linear regression predictions using the multivariate normal assumption.

    Parameters:
    X (numpy.ndarray): The design matrix of shape (n, k_plus_1), where n is the number of observations and k_plus_1 is the number of predictors plus one.
    y (numpy.ndarray): The response variable of shape (n,).
    D (numpy.ndarray): The matrix of predictors for which predictions are desired, of shape (m, k_plus_1).
    alpha (float): The significance level for the confidence intervals.
    method (str, optional): The method to determine the critical value. Can be "best" (default), "Bonferroni", or "Scheffe".

    Returns:
    numpy.ndarray: The confidence intervals for the predictions, of shape (m, 2).

    """
    # Number of observations and predictors
    n, k_plus_1 = X.shape
    
    XTX_inv = np.linalg.inv(X.T @ X)

    # Compute beta_hat (MLE for beta)
    beta_hat = XTX_inv @ X.T @ y
    
    # Compute the unbiased estimate for sigma^2
    residuals = y - X @ beta_hat
    Se_2 = (residuals.T @ residuals) / (n - k_plus_1)
    
    S_e = np.sqrt((residuals.T @ residuals) / (n - k_plus_1))
    
    # Number of predictions
    m = D.shape[0]
    
    bonferroni_critical_value = t.ppf(1 - alpha / (2 * m), df=n - k_plus_1)
    
    scheffe_critical_value = np.sqrt(k_plus_1 * f.ppf(1 - alpha, dfn=k_plus_1, dfd=n - k_plus_1))

    if method == "Best":
        if bonferroni_critical_value < scheffe_critical_value:
            critical_value = bonferroni_critical_value
            print("Bonferroni is better")
        else:
            critical_value = scheffe_critical_value 
            print("Scheffe is better")   
    elif method == "Bonferroni":
        critical_value = bonferroni_critical_value
    elif method == "Scheffe":
        critical_value = scheffe_critical_value
    else:
        print("Invalid method. Using the best method.")
        if bonferroni_critical_value < scheffe_critical_value:
            critical_value = bonferroni_critical_value
        else:
            critical_value = scheffe_critical_value

    
    # Compute the confidence intervals
    standart_errors = S_e * np.sqrt(np.diag(D @ XTX_inv @ D.T))
    predictions = D @ beta_hat

    ci = np.zeros((m, 2))
    for i in range(m):
        ci[i, 0] = predictions[i] - critical_value * standart_errors[i]
        ci[i, 1] = predictions[i] + critical_value * standart_errors[i]
    return ci
        

