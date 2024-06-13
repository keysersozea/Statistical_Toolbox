
import statistical_toolbox as st
import numpy as np
import pandas as pd

def main():
    
    alpha = 0.05
    
    # ANOVA 1 Inputs
    data_1 = [
    [23, 20, 22, 21, 24, 26, 25, 27, 22, 20],
    [24, 24, 24, 28, 24.8, 25, 26, 27, 24, 24.9],
    [31, 30, 32, 33, 29, 31, 30, 32, 33, 29],
    [35, 34, 36, 35, 34, 36, 35, 34, 36, 35]
    ]
        
    C_1 = np.array([
        [1, -1, 0, 0],  
        [0, 1, -1, 0], 
        [0, 0, 1, -1],  
        [1, 0, -1, 0],  
        [1, 0, 0, -1]   
    ])
    
    d = np.zeros(C_1.shape[0])
    
    # ANOVA 2 Inputs
    X = np.array([
    [[10, 12, 11, 13, 14, 15], [20, 22, 21, 23, 24, 25], [30, 32, 31, 33, 34, 35], [40, 42, 41, 43, 44, 45]],
    [[11, 13, 12, 14, 15, 16], [21, 23, 22, 24, 25, 26], [31, 33, 32, 34, 35, 36], [41, 43, 42, 44, 45, 46]],
    [[12, 14, 13, 15, 16, 17], [22, 24, 23, 25, 26, 27], [32, 34, 33, 35, 36, 37], [42, 44, 43, 45, 46, 47]],
    [[13, 15, 14, 16, 17, 18], [23, 25, 24, 26, 27, 28], [33, 35, 34, 36, 37, 38], [43, 45, 44, 46, 47, 48]],
    [[14, 16, 15, 17, 18, 19], [24, 26, 25, 27, 28, 29], [34, 36, 35, 37, 38, 39], [44, 46, 45, 47, 48, 49]]
    ])


    # Linear Regression Inputs
    
    X_lr = np.array([
    [1, 42.8, 40.0], [1, 63.5, 93.5], [1, 37.5, 35.5], [1, 39.5, 30.0], 
    [1, 45.5, 52.0], [1, 38.5, 17.0], [1, 43.0, 38.5], [1, 22.5, 8.5], 
    [1, 37.0, 33.0], [1, 23.5, 9.5], [1, 33.0, 21.0], [1, 58.0, 79.0],
    [1, 47.8, 42.0], [1, 66.5, 96.5], [1, 40.5, 38.5], [1, 42.5, 33.0], 
    [1, 48.5, 55.0], [1, 41.5, 20.0], [1, 46.0, 41.5], [1, 25.5, 11.5], 
    [1, 40.0, 36.0], [1, 26.5, 12.5], [1, 36.0, 24.0], [1, 61.0, 82.0],
    [1, 50.8, 45.0], [1, 69.5, 99.5], [1, 43.5, 41.5], [1, 45.5, 36.0], 
    [1, 51.5, 58.0], [1, 44.5, 23.0], [1, 49.0, 44.5], [1, 28.5, 14.5], 
    [1, 43.0, 39.0], [1, 29.5, 15.5], [1, 39.0, 27.0], [1, 64.0, 85.0],
    [1, 53.8, 48.0], [1, 72.5, 102.5], [1, 46.5, 44.5], [1, 48.5, 39.0], 
    [1, 54.5, 61.0], [1, 47.5, 26.0], [1, 52.0, 47.5], [1, 31.5, 17.5], 
    [1, 46.0, 42.0], [1, 32.5, 18.5], [1, 42.0, 30.0], [1, 67.0, 88.0]
    ])
    y_lr = np.array([
        37.0, 49.5, 34.5, 36.0, 43.0, 28.0, 37.0, 20.0, 33.5, 30.5, 38.5, 47.0,
        40.0, 52.5, 37.5, 39.0, 46.0, 31.0, 40.0, 23.0, 36.0, 33.0, 41.0, 50.0,
        43.0, 55.5, 40.5, 42.0, 49.0, 34.0, 43.0, 26.0, 39.0, 35.5, 43.5, 53.0,
        46.0, 58.5, 43.5, 45.0, 52.0, 37.0, 46.0, 29.0, 42.0, 38.0, 46.0, 56.0
    ])
    D_lr = np.array([
        [1, 30.0, 30.0], [1, 40.0, 40.0], [1, 50.0, 50.0], 
        [1, 60.0, 60.0], [1, 70.0, 70.0], [1, 80.0, 80.0]
    ])
    
    C_lr = np.array([
    [0, 1, -1],  # Test if beta1 - beta2 = c0_1
    [0, 0, 1]   # Test if beta2 = c0_2"
    ])
    
    c0 = np.zeros(C_lr.shape[0])
    
    indices = np.array([1, 2]) # Test if beta_1 = beta_3 = 0
    

    ### ANOVA_1 ###
    print("\n")
    print("\n##############################")
    print("##############################\n")
    print("ANOVA_1\n")
    print("##############################")
    print("##############################\n")


    ### PRESENT DATA ###
    print("\n ############################## \n")
    df_data = pd.DataFrame(data_1).T.rename(columns={0: "Group 1", 1: "Group 2", 2: "Group 3", 3: "Group 4"})
    print("Data:")
    print(df_data)
    
    ### PRESENT COEFFIENT MATRIX ###
    print("\n ############################## \n")
    print("Coefficient Matrix:")
    print(C_1)
    
    
    ### ANOVA1_partition_TSS ###
    print("\n ############################## \n")
    print("Testing ANOVA1_partition_TSS ...\n")
    SS_total, SS_within, SS_between = st.ANOVA1_partition_TSS(data_1)
    
    ### ANOVA1_test_equality ###
    print("\n ############################## \n")
    print("Testing ANOVA1_test_equality ...\n")
    critical_value, p_value, reject_null = st.ANOVA1_test_equality(data_1)
    
    ### ANOVA1_is_contrast ###
    print("\n ############################## \n")
    print("Testing ANOVA1_is_contrast ...\n")
    is_contrast = st.ANOVA1_is_contrast(C_1)
    print("Is contrast: ", is_contrast)
    
    ### ANOVA1_is_orthogonal ###
    print("\n ############################## \n")
    print("Testing ANOVA1_is_orthogonal ...\n")
    num_groups = len(data_1)
    num_obs_per_group = [len(group) for group in data_1]
    is_orthogonal = st.ANOVA1_is_orthogonal(num_obs_per_group, C_1[0, :], C_1[1, :])
    print("Is orthogonal: ", is_orthogonal)
    print('Note: It is only demonstrated for the first two contrasts.')
    
    ### Bonferroni_correction ###
    print("\n ############################## \n")
    print("Testing Bonferroni_correction ...\n")
    bonferroni_alpha = st.Bonferroni_correction(alpha, C_1.shape[0])
    print("alpha: ", alpha)
    print("Bonferroni alpha: ", bonferroni_alpha)
    
    ### Sidak_correction ###
    print("\n ############################## \n")
    print("Testing Sidak_correction ...\n")
    sidak_alpha = st.Sidak_correction(alpha, C_1.shape[0])
    print("alpha: ", alpha)
    print("Sidak alpha: ", sidak_alpha)    

    
    ### ANOVA1_CI_linear_combs ###
    print("\n ############################## \n")
    print("##############################\n")
    print("Testing ANOVA1_CI_linear_combs ...\n")
    print("Scheffe's CI:")
    schefffe_CI = st.ANOVA1_CI_linear_combs(data_1, C_1, alpha, method="Scheffe")
    print(schefffe_CI)
    print("\n ############################## \n")
    print("Tukey's CI:")
    tukey_CI = st.ANOVA1_CI_linear_combs(data_1, C_1, alpha, method="Tukey")
    print(tukey_CI)
    print("\n ############################## \n")
    print("Bonferroni's CI:")
    bonferroni_CI = st.ANOVA1_CI_linear_combs(data_1, C_1, alpha, method="Bonferroni")
    print(bonferroni_CI)
    print("\n ############################## \n")
    print("Sidak's CI:")
    sidak_CI = st.ANOVA1_CI_linear_combs(data_1, C_1, alpha, method="Sidak")
    print(sidak_CI)
    print("\n ############################## \n")
    print("Best CI:")
    best_CI = st.ANOVA1_CI_linear_combs(data_1, C_1, alpha, method="Best")
    print(best_CI)
    
    
    ### ANOVA1_test_linear_combs ###
    print("\n ############################## \n")
    print("############################## \n")
    print("Testing ANOVA1_test_linear_combs ...\n")
    print("Scheffe's test:")
    scheffe_p_value, scheffe_reject_null = st.ANOVA1_test_linear_combs(data_1, C_1, alpha, method="Scheffe")
    print("p-value: ", scheffe_p_value)
    print("Reject null: ", scheffe_reject_null)
    print("\n ############################## \n")
    print("Tukey's test:")
    tukey_p_value, tukey_reject_null = st.ANOVA1_test_linear_combs(data_1, C_1, alpha, method="Tukey")
    print("p-value: ", tukey_p_value)
    print("Reject null: ", tukey_reject_null)
    print("\n ############################## \n")
    print("Bonferroni's test:")
    bonferroni_p_value, bonferroni_reject_null = st.ANOVA1_test_linear_combs(data_1, C_1, alpha, method="Bonferroni")
    print("p-value: ", bonferroni_p_value)
    print("Reject null: ", bonferroni_reject_null)
    print("\n ############################## \n")
    print("Sidak's test:")
    sidak_p_value, sidak_reject_null = st.ANOVA1_test_linear_combs(data_1, C_1, alpha, method="Sidak")
    print("p-value: ", sidak_p_value)
    print("Reject null: ", sidak_reject_null)
    print("\n ############################## \n")
    print("Best test:")
    best_p_value, best_reject_null = st.ANOVA1_test_linear_combs(data_1, C_1, alpha, method="Best")
    print("p-value: ", best_p_value)
    print("Reject null: ", best_reject_null )
    

    
    
    ### ANOVA_2 ###
    print("\n")
    print("\n")
    print("\n##############################")
    print("##############################\n")
    print("ANOVA_2\n")
    print("##############################")
    print("##############################\n")
    
    
    # Present data #
    print("\n ############################## \n")
    print("Data:")
    print(X)

    # Demonstrate ANOVA2_partition_TSS
    print("\n ############################## \n")
    print("Testing ANOVA2_partition_TSS...\n")
    SS_total, SSA, SSB, SSAB, SSE = st.ANOVA2_partition_TSS(X)
    print("SS_total:", SS_total)
    print("SSA:", SSA)
    print("SSB:", SSB)
    print("SSAB:", SSAB)
    print("SSE:", SSE)

    # Demonstrate ANOVA2_MLE
    print("\n ############################## \n")
    print("Testing ANOVA2_MLE...\n")
    mu, ai, bj, delta_ij = st.ANOVA2_MLE(X)
    print("mu:", mu)
    print("ai:", ai)
    print("bj:", bj)
    print("delta_ij:", delta_ij)

    # Demonstrate ANOVA2_test_equality
    print("############################## \n")
    print("\nTesting ANOVA2_test_equality...")
    print("\nTest result for 'A':")
    test_result_A = st.ANOVA2_test_equality(X, alpha, 'A')
    print("\nTest result for 'B':")
    test_result_B = st.ANOVA2_test_equality(X, alpha, 'B')
    print("\nTest result for 'AB':")
    test_result_AB = st.ANOVA2_test_equality(X, alpha, 'AB')
    
    
    
    
    ### Linear Regression ###
    
    print("\n")
    print("\n##############################")
    print("##############################\n")
    print("Linear Regression\n")
    print("##############################")
    print("##############################\n")
    
    
    ### PRESENT DATA ###
    print("\n ############################## \n")
    print("X:")
    print(X_lr)
    print("\n ############################## \n")
    print("y:")
    print(y_lr)
    print("\n ############################## \n")
    print("D:")
    print(D_lr)
    print("\n ############################## \n")
    print("C:")
    print(C_lr)
    print("\n ############################## \n")
    print("c0:")
    print(c0)
    print("\n ############################## \n")
    print("Indices used in Mult_norm_LR_test_comp function:")
    print(indices)
    
    ### Mult_LR_Least_squares ###
    print("\n ############################## \n")
    print("Testing Mult_LR_Least_squares ...\n")
    beta_hat, sigma2_hat_MLE, sigma2_hat_unbiased = st.Mult_LR_Least_squares(X_lr, y_lr)
    print("Beta hat:")
    print(beta_hat)
    print("\nSigma2 hat MLE:")
    print(sigma2_hat_MLE)
    print("\nSigma2 hat unbiased:")
    print(sigma2_hat_unbiased)
    
    ### Mult_LR_partition_TSS ###
    print("\n ############################## \n")
    print("Testing Mult_LR_partition_TSS ...\n")
    TSS, RegSS, RSS = st.Mult_LR_partition_TSS(X_lr, y_lr)
    print("TSS:")
    print(TSS)
    print("\nRegSS:")
    print(RegSS)
    print("\nRSS:")
    print(RSS)
    
    ### Mult_norm_LR_simul_CI ###
    print("\n ############################## \n")
    print("Testing Mult_norm_LR_simul_CI ...\n")
    CI = st.Mult_norm_LR_simul_CI(X_lr, y_lr, alpha)
    print("CI:")
    print(CI)
    
    ### Mult_norm_LR_CR ###
    print("\n ############################## \n")
    print("Testing Mult_norm_LR_CR ...\n")
    center, shape_matrix, radius = st.Mult_norm_LR_CR(X_lr, y_lr, C_lr, alpha)
    print("Center:")
    print(center)
    print("\nShape matrix:")
    print(shape_matrix)
    print("\nRadius:")
    print(radius)
    
    ### Mult_norm_LR_is_in_CR ###
    print("\n ############################## \n")
    print("Testing Mult_norm_LR_is_in_CR ...\n")
    is_in_CR = st.Mult_norm_LR_is_in_CR(X_lr, y_lr, C_lr, c0, radius)
    print("Is in CR:")
    print(is_in_CR)
    
    ### Mult_norm_LR_test_general ###
    print("\n ############################## \n")
    print("Testing Mult_norm_LR_test_general ...\n")
    reject_null_general = st.Mult_norm_LR_test_general(X_lr, y_lr, C_lr, c0, alpha)
    print("Reject null:")
    print(reject_null_general)
    
    ### Mult_norm_LR_test_comp ###
    print("\n ############################## \n")
    print("Testing Mult_norm_LR_test_comp ...\n")
    reject_null_comp = st.Mult_norm_LR_test_comp(X_lr, y_lr, indices, alpha)
    print("Reject null:")
    print(reject_null_comp)

    ### Mult_norm_LR_test_linear_reg ###
    print("\n ############################## \n")
    print("Testing Mult_norm_LR_test_linear_reg ...\n")
    reject_null_linear_reg = st.Mult_norm_LR_test_linear_reg(X_lr, y_lr, alpha)
    print("Reject null:")
    print(reject_null_linear_reg)
    
    ### Mult_norm_LR_pred_CI ###
    print("\n ############################## \n")
    print("Testing Mult_norm_LR_pred_CI ...\n")
    print("Scheffe's CI:")
    LR_schefffe_CI = st.Mult_norm_LR_pred_CI(X_lr, y_lr, D_lr, alpha, method="Scheffe")
    print(LR_schefffe_CI)
    print("\n ############################## \n")
    print("Bonferroni's CI:")
    LR_bonferroni_CI = st.Mult_norm_LR_pred_CI(X_lr, y_lr, D_lr, alpha, method="Bonferroni")
    print(LR_bonferroni_CI)
    print("\n ############################## \n")
    print("Best CI:")
    LR_best_CI = st.Mult_norm_LR_pred_CI(X_lr, y_lr, D_lr, alpha, method="Best")
    print(LR_best_CI)
    
    print("\n ############################## \n")
    print("END\n")
    
    
if __name__ == "__main__":
    main()