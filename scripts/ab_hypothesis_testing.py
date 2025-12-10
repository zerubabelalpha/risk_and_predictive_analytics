import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any

class ABHypothesisTester:
    """
    A class for performing A/B hypothesis testing on insurance data.
    
    Metrics:
    - Claim Frequency: Proportion of policies with at least one claim.
    - Claim Severity: Average claim amount given a claim occurred.
    - Margin: TotalPremium - TotalClaims.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with the dataframe.
        
        Args:
            df: Cleaned DataFrame containing insurance data.
        """
        self.df = df.copy()
        # Ensure necessary columns exist or are calculable
        if 'Margin' not in self.df.columns:
            if 'TotalPremium' in self.df.columns and 'TotalClaims' in self.df.columns:
                self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
            else:
                print("Warning: 'Margin' could not be calculated. key columns missing.")

        self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)

    def calculate_metrics(self, group_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate key metrics for a given group.
        """
        total_policies = len(group_df)
        if total_policies == 0:
            return {"Frequency": 0, "Severity": 0, "Margin": 0}
            
        # Claim Frequency
        claim_count = group_df['HasClaim'].sum()
        frequency = claim_count / total_policies
        
        # Claim Severity (avg claim amount for those who claimed)
        claims_only = group_df[group_df['HasClaim'] == 1]
        severity = claims_only['TotalClaims'].mean() if len(claims_only) > 0 else 0.0
        
        # Margin
        avg_margin = group_df['Margin'].mean() if 'Margin' in group_df.columns else 0.0
        
        return {
            "Frequency": frequency,
            "Severity": severity,
            "Margin": avg_margin,
            "PolicyCount": total_policies,
            "ClaimCount": claim_count
        }

    def perform_chi_squared(self, group_a: pd.DataFrame, group_b: pd.DataFrame, target_col: str, group_col_name: str, a_label: str, b_label: str) -> Dict[str, Any]:
        """
        Perform Chi-Squared test for categorical/proportion differences (e.g., Claim Frequency).
        Target col should be binary (0/1).
        """
        # Contingency table:
        #           Claim   No Claim
        # Group A    n_Ac    n_An
        # Group B    n_Bc    n_Bn
        
        n_A_claim = group_a[target_col].sum()
        n_A_no_claim = len(group_a) - n_A_claim
        
        n_B_claim = group_b[target_col].sum()
        n_B_no_claim = len(group_b) - n_B_claim
        
        contingency_table = [
            [n_A_claim, n_A_no_claim],
            [n_B_claim, n_B_no_claim]
        ]
        
        chi2, p_value, dof, ex = stats.chi2_contingency(contingency_table)
        
        return {
            "TestType": "Chi-Squared",
            "Statistic": chi2,
            "P-Value": p_value,
            "Significant": p_value < 0.05,
            "GroupA_Mean": n_A_claim / len(group_a),
            "GroupB_Mean": n_B_claim / len(group_b)
        }

    def perform_t_test(self, group_a: pd.DataFrame, group_b: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Perform T-Test for numerical mean differences (e.g., Margin).
        Assumes independent samples.
        """
        data_a = group_a[target_col].dropna()
        data_b = group_b[target_col].dropna()
        
        t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False) # Welch's t-test
        
        return {
            "TestType": "T-Test (Welch)",
            "Statistic": t_stat,
            "P-Value": p_value,
            "Significant": p_value < 0.05,
            "GroupA_Mean": data_a.mean(),
            "GroupB_Mean": data_b.mean()
        }
        
    def perform_mann_whitney(self, group_a: pd.DataFrame, group_b: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """
        Perform Mann-Whitney U Test for numerical differences (e.g., Severity).
        Better for skewed distributions like insurance claims.
        """
        data_a = group_a[target_col].dropna()
        data_b = group_b[target_col].dropna()
        
        u_stat, p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        
        return {
            "TestType": "Mann-Whitney U",
            "Statistic": u_stat,
            "P-Value": p_value,
            "Significant": p_value < 0.05,
            "GroupA_Median": data_a.median(),
            "GroupB_Median": data_b.median(),
            "GroupA_Mean": data_a.mean(),
            "GroupB_Mean": data_b.mean()
        }

    def test_hypothesis(self, feature_col: str, group_a_val: Any, group_b_val: Any, 
                        metric_type: str = "Risk",  # "Risk" (Frequency & Severity) or "Margin"
                        segment_col_override: str = None
                       ) -> Dict[str, Any]:
        """
        Generalized method to test hypotheses.
        
        Args:
            feature_col: Column to split groups by (e.g., 'Province', 'Gender').
            group_a_val: Value(s) defining Group A.
            group_b_val: Value(s) defining Group B.
            metric_type: "Risk" or "Margin".
        """
        col = segment_col_override if segment_col_override else feature_col
        
        # Segment Data
        if isinstance(group_a_val, list):
            group_a = self.df[self.df[col].isin(group_a_val)]
        else:
            group_a = self.df[self.df[col] == group_a_val]
            
        if isinstance(group_b_val, list):
            group_b = self.df[self.df[col].isin(group_b_val)]
        else:
            group_b = self.df[self.df[col] == group_b_val]
            
        results = {}
        
        if metric_type == "Risk":
            # 1. Claim Frequency (Chi-Squared)
            res_freq = self.perform_chi_squared(group_a, group_b, 'HasClaim', feature_col, str(group_a_val), str(group_b_val))
            results['Frequency_Test'] = res_freq
            
            # 2. Claim Severity (Mann-Whitney or T-test) - Only for those who had claims
            g_a_claims = group_a[group_a['HasClaim'] == 1]
            g_b_claims = group_b[group_b['HasClaim'] == 1]
            
            if len(g_a_claims) > 5 and len(g_b_claims) > 5:
                res_sev = self.perform_t_test(g_a_claims, g_b_claims, 'TotalClaims') # Or Mann-Whitney
                results['Severity_Test'] = res_sev
            else:
                results['Severity_Test'] = {"Error": "Insufficient sample size for severity test"}
                
        elif metric_type == "Margin":
            # Margin (T-Test)
            res_margin = self.perform_t_test(group_a, group_b, 'Margin')
            results['Margin_Test'] = res_margin
            
        return results

    def plot_results(self, test_results: Dict[str, Any], group_a_label: str, group_b_label: str, title_suffix: str = ""):
        """
        Visualize the testing results.
        """
        # Setup plotting based on what results we have
        metrics_to_plot = []
        if 'Frequency_Test' in test_results:
            metrics_to_plot.append(('Frequency', test_results['Frequency_Test']))
        if 'Severity_Test' in test_results and 'Error' not in test_results['Severity_Test']:
            metrics_to_plot.append(('Severity', test_results['Severity_Test']))
        if 'Margin_Test' in test_results:
            metrics_to_plot.append(('Margin', test_results['Margin_Test']))
            
        if not metrics_to_plot:
            print("No valid results to plot.")
            return

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 5))
        if len(metrics_to_plot) == 1:
            axes = [axes]
            
        for i, (metric, res) in enumerate(metrics_to_plot):
            ax = axes[i]
            
            # Data for plotting
            means = [res['GroupA_Mean'], res['GroupB_Mean']]
            labels = [group_a_label, group_b_label]
            colors = ['#3498db', '#e74c3c']
            
            bars = ax.bar(labels, means, color=colors, alpha=0.7)
            
            # Add p-value annotation
            p_val = res['P-Value']
            significance = "Significant" if res['Significant'] else "Not Significant"
            ax.set_title(f"{metric} Comparison\n(p={p_val:.4f}, {significance})")
            ax.set_ylabel(f"Mean {metric}")
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom')
                        
        plt.suptitle(f"A/B Testing Results: {title_suffix}", fontsize=14)
        plt.tight_layout()
        plt.show()

# --- Wrapper functions for specific requested hypotheses ---

    def test_risk_provinces(self, province_a: str, province_b: str):
        """
        H0: There are no risk differences across provinces (A vs B).
        """
        print(f"\n--- Testing Risk Differences: {province_a} vs {province_b} ---")
        res = self.test_hypothesis('Province', province_a, province_b, "Risk")
        self.plot_results(res, province_a, province_b, "Risk by Province")
        return res

    def test_risk_zipcodes(self, zip_col: str, zip_a, zip_b):
        """
        H0: There are no risk differences between zip codes.
        """
        print(f"\n--- Testing Risk Differences: Zip {zip_a} vs {zip_b} ---")
        res = self.test_hypothesis(zip_col, zip_a, zip_b, "Risk")
        self.plot_results(res, f"Zip {zip_a}", f"Zip {zip_b}", "Risk by Zip Code")
        return res

    def test_margin_zipcodes(self, zip_col: str, zip_a, zip_b):
        """
        H0: There is no significant margin (profit) difference between zip codes.
        """
        print(f"\n--- Testing Margin Differences: Zip {zip_a} vs {zip_b} ---")
        res = self.test_hypothesis(zip_col, zip_a, zip_b, "Margin")
        self.plot_results(res, f"Zip {zip_a}", f"Zip {zip_b}", "Margin by Zip Code")
        return res

    def test_risk_gender(self, gender_col: str = "Gender", val_women: str = "Female", val_men: str = "Male"):
        """
        H0: There is no significant risk difference between Women and Men.
        """
        print(f"\n--- Testing Risk Differences: {val_women} vs {val_men} ---")
        res = self.test_hypothesis(gender_col, val_women, val_men, "Risk")
        self.plot_results(res, val_women, val_men, "Risk by Gender")
        return res
