import numpy as np
import itertools
import os
from collections import defaultdict
import time
from .utils import *
from functools import reduce

from scipy.sparse import csr_matrix
import pandas as pd
from fractions import Fraction

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def find_rational_approximation(x, max_denominator=1000):
    """
    Find the best rational approximation of a number using the Fraction class.
    max_denominator limits the size of the denominator to keep the fraction manageable.
    """
    return Fraction(x).limit_denominator(max_denominator)

def rational_approx_sqrt(x, max_denominator=100):
    """
    Find the best rational approximation s/t for sqrt(x).
    Returns (s, t) as integers.
    """
    sqrt_x = np.sqrt(x)
    frac = Fraction(sqrt_x).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator
"""
domains = [3, 3, 3 ,3 ,3]
5 attributes, each attribute 3 values
-------------------------------------------
att = ()        --> sum query
att = (0)       --> marginal A
att = (2, 3)    --> marginal CD
att = (0, 1 ,2) --> marginal ABC
"""


class MargMech:

    def __init__(self, domains, att, var_bound=1.0):
        self.domains = domains
        self.num_att = len(domains)
        self.att = att
        self.var_bound = var_bound
        self.covar = None
        self.noisy_answer_vector = None
        cur_domains = [self.domains[at] for at in att]
        self.num_queries = np.prod([c + 0.0 for c in cur_domains])
        self.variance = None
        self.non_noisy_vector = None
        self.csv_data = None
        pass

    def output_bound(self):
        return self.var_bound

    def input_noisy_answer_vector(self, answer):
        self.noisy_answer_vector = answer

    def input_non_noisy_vector(self, answer):
        self.non_noisy_vector = answer
    
    def input_csv_data(self, data):
        self.csv_data = data

    def get_csv_data(self):
        return self.csv_data

    def get_num_queries(self):
        return self.num_queries

    def input_variance(self, var):
        self.variance = var

    def output_variance(self):
        return self.variance

    def get_noisy_answer_vector(self):
        return self.noisy_answer_vector

    def get_non_noisy_vector(self):
        return self.non_noisy_vector

    def input_covar_result(self, phi_result):
        self.phi_result=phi_result

    def input_trace_result(self, trace_result):
        self.trace_result=trace_result

    def input_identity_cov_result(self, identity_cov_result):
        self.identity_cov_result=identity_cov_result

    def input_identity_trace_result(self, identity_trace_result):
        self.identity_trace_result=identity_trace_result

    def get_covar_result(self):
        return self.phi_result

    def get_trace_result(self):
        return self.trace_result
    
    def get_identity_cov_result(self):
        return self.identity_cov_result

    def get_identity_trace_result(self):
        return self.identity_trace_result

    pass


class ResMech:

    def __init__(self, domains, att, residual_matrix, gamma_matrix=None):
        self.col_names = None
        self.data = None
        self.domains = domains
        self.num_att = len(self.domains)
        self.att = att
        self.core_mat = None
        self.res_mat_list = []
        self.gamma_mat_list = []
        self.get_core_matrix(residual_matrix, gamma_matrix)

        self.noise_level = None
        self.covar = None
        self.calculated = False
        self.recon_answer = None
        self.noisy_answer_vector = None
        self.non_noisy_vector = None
        self.true_recon_answer = None
        self.csv_data = None
    #change the sub matrix to identity matrix
    def get_core_matrix(self, residual_matrix, gamma_matrix=None, using_identity=False):
        att_set = set(list(self.att))
        for i in range(0, self.num_att):
            att_size = self.domains[i]
            if i in att_set:
                if using_identity:
                    res_mat = subtract_matrix(att_size)
                    gamma_mat = subtract_matrix(att_size)
                else:
                    res_mat = residual_matrix[att_size]
                    gamma_mat = gamma_matrix[att_size] if gamma_matrix else res_mat
                self.res_mat_list.append(res_mat)
                self.gamma_mat_list.append(gamma_mat)

    def input_noise_level(self, noise_level):
        self.noise_level = noise_level
        self.calculated = True

    def output_noise_level(self):
        if self.calculated:
            return self.noise_level
        else:
            print("Not yet calculated!")
            return 0.0

    def is_calculated(self):
        return self.calculated

    def input_data(self, data, col_names):
        self.data = data
        self.col_names = col_names
        pass

    def measure(self, input_base=None, debug=False, n_runs=None):
        sub_domains = [self.domains[at]+0.0 for at in self.att]
        bins = [np.arange(t+1) for t in sub_domains]
        if self.att == ():
            sparse_vec = np.array(len(self.data))
        else:
            datavector = np.histogramdd(self.data.values, bins)[0]
            datavector = datavector.flatten()
            sparse_vec = datavector
        non_noisy_vector = mult_kron_vec(self.res_mat_list, sparse_vec)
        col_size = np.prod(sub_domains).astype(int)

        gamma_col_size = int(np.prod([g.shape[1] for g in self.gamma_mat_list])) if self.gamma_mat_list else col_size

        if debug==True and (n_runs is not None and n_runs > 1):
            cov_sum = None
            for _ in range(n_runs):
                rd = np.sqrt(self.noise_level) * np.random.normal(size=[gamma_col_size, 1])
                cov_rd = mult_kron_vec(self.gamma_mat_list, rd)
                cov = cov_rd @ cov_rd.T
                if cov_sum is None:
                    cov_sum = np.zeros_like(cov)
                cov_sum += cov
            avg_cov = cov_sum / n_runs
            # For compatibility, still set noisy/non_noisy/csv_data as in the single run
            rd = np.sqrt(self.noise_level) * np.random.normal(size=[gamma_col_size, 1])
            cov_rd = mult_kron_vec(self.gamma_mat_list, rd)
            self.noisy_answer_vector = non_noisy_vector + cov_rd
            self.non_noisy_vector = non_noisy_vector + np.zeros_like(cov_rd)
            self.csv_data = sparse_vec.reshape(-1,1)
            return avg_cov
        else:
            rd = np.sqrt(self.noise_level) * np.random.normal(size=[gamma_col_size, 1])
            cov_rd = mult_kron_vec(self.gamma_mat_list, rd)
            self.noisy_answer_vector = non_noisy_vector + cov_rd
            self.non_noisy_vector = non_noisy_vector + np.zeros_like(cov_rd)
            self.csv_data = sparse_vec.reshape(-1,1)
            return None

    def get_recon_answer(self, mat_list):
        self.recon_answer = mult_kron_vec(mat_list, self.noisy_answer_vector)
        return self.recon_answer

    def get_origin_answer(self, mat_list):
        self.true_recon_answer = mult_kron_vec(mat_list, self.non_noisy_vector)
        return self.true_recon_answer
    
    def get_csv_data(self):
        return self.csv_data

    def measure_gaussian(self, n_runs, domains, debug=False):
        """
        For a zero dataset, for each measurement, generate z ~ N(0, r^2 I) where
        r^2 = s^2 * |Atti|^2 for all atti in the measurement, with s/t ~ sqrt(noise_level).
        Return and average zz^T over n_runs for each query.
        """
        att_sizes = [domains[i] for i in self.att]
        if len(att_sizes) == 0:
            return None
        res_list = [mat / domains[i] for mat, i in zip(self.res_mat_list, self.att)]
        s, t = rational_approx_sqrt(self.noise_level)
        r2 = (s/t)**2 * np.prod([size**2 for size in att_sizes])
        col_size = np.prod(att_sizes)
        cov_sum = None
        z_samples = [] if debug else None
        z1_samples = [] if debug else None
        for _ in range(n_runs):
            z = np.sqrt(r2) * np.random.normal(size=[col_size, 1])
            z1 = mult_kron_vec(res_list, z)
            cov = z1 @ z1.T
            if cov_sum is None:
                cov_sum = np.zeros_like(cov)
            cov_sum += cov
            if debug:
                z_samples.append(z)
                z1_samples.append(z1)
        avg_cov = cov_sum / n_runs
        if debug:
            return avg_cov, z_samples, z1_samples
        else:
            return avg_cov


class ResidualPlanner:

    def __init__(self, domains, bases=None, subtract_version='v1'):
        self.col_names = None
        self.data = None
        self.domains = domains
        self.bases = bases
        self.num_of_mech = 0
        self.num_of_res = 0

        self.marg_index = {}
        self.res_index = {}
        self.id2res = {}

        self.marg_dict = {}
        self.res_dict = {}

        self.pcost_coeff = {}
        self.var_coeff = {}
        self.var_bound = {}
        self.var_coeff_sum = defaultdict(int)

        self.sparse_row = {}
        self.sparse_col = {}

        self.pcost_sum = {}
        self.pcost_res = {}
        self.var_sum = {}
        self.var_res = {}

        self.residual_matrix = {}
        self.gamma_matrix = {}
        self.residual_pinv = {}
        self.one_mat = {}
        self.subtract_version = subtract_version

    def preprocessing(self):
        pass

    def input_mech(self, att):
        pass

    def input_data(self, data, col_names):
        self.data = data
        self.col_names = col_names
        pass

    def output_coefficient(self):
        pass

    def get_noise_level(self):
        pass

    def measurement(self):
        #print("Start Measurement, total number of cells: ", len(self.res_dict))
        for i, att in enumerate(self.res_dict.keys()):
            #if i % 10_000 == 0 and i > 0:
                #print("Measuring cell: ", i)
            res_mech = self.res_dict[att]
            cols = [self.col_names[idx] for idx in att]
            sub_data = self.data.loc[:, cols]
            input_base=[self.bases[idx] for idx in att]
            res_mech.input_data(sub_data, cols)
            res_mech.measure(input_base)

    def reconstruction(self, debug=False):
        #print("Start Reconstruction, total number of queries: ", len(self.marg_dict))
        debug_results = {} if debug else None
        for i, att in enumerate(self.marg_dict.keys()):
            #if i % 10_000 == 0 and i > 0:
                #print("Reconstructing marginal: ", i)
            mech = self.marg_dict[att]
            
            # First calculate the Kronecker product for all attributes in att
            att_mat_list = []
            att_mat_dict = {}
            for at in att:
                base = self.bases[at]
                if base == 'P':
                    # For prefix basis
                    work = prefix_workload(self.domains[at])
                elif base == 'I':
                    # For identity basis
                    work = np.eye(self.domains[at])
                elif base == 'R':
                    # For range basis
                    work = range_workload(self.domains[at])
                elif base == 'H':
                    # For Haar basis, use identity as the workload
                    work = np.eye(self.domains[at])
                att_mat_list.append(work)
                att_mat_dict[at] = work
            # Calculate Kronecker product using np.kron if we have matrices
            # if len(att_mat_list) > 0:
            #     # Start with first matrix
            #     kron_result = att_mat_list[0]
            #     # Sequentially compute Kronecker product with remaining matrices
            #     for j in range(1, len(att_mat_list)):
            #         kron_result = np.kron(kron_result, att_mat_list[j])
            
            att_subsets = all_subsets(att)
            noisy_answer_vector = 0.0
            non_noisy_vector = 0.0
            csv_data = []
            covar_result=[]
            trace_result=[]
            identity_cov_result=[]
            identity_trace_result=[]
            
            #for each A' in closure(workload)
            #TODO get noisey level of A'
            for subset in att_subsets:
                res_mech = self.res_dict[subset]
                mat_list = []
                phi_list=[]
                product_A = 1.0
                product_Abar=1.0
                identity_product_A=1.0
                identity_product_B=1.0
                identity_product_Abar=1.0
                

                # For each attribute in the target marginal
                #Format this one with different therome
                for at in att:
                    if at in subset:
                        # If attribute is in subset, use residual matrix

                        # need to check in the case of identity matrix, it will use subtract_matrix
                        #sub_mat = self.residual_matrix[self.domains[at]]
                        #sub_mat=subtract_matrix(self.domains[at])
                        #TODO put before the reconstruction
                        sub_pinv = self.residual_pinv[self.domains[at]]
                        mat_list.append(sub_pinv)
                        # Debug mode: collect phi, product_A, etc.
                        if debug:
                            # Example: phi = att_mat_dict[at] @ sub_pinv
                            phi = att_mat_dict[at] @ sub_pinv
                            phi_Total = phi @ phi.T
                            phi_list.append(phi_Total)
                            product_A = product_A * np.linalg.norm(phi, 'fro') ** 2
                            identity_product_A = identity_product_A * ((self.domains[at]-1)/self.domains[at])
                            identity_product_B = identity_product_B * (-1/self.domains[at])
                    else:
                        one_mat = self.one_mat[self.domains[at]]
                        mat_list.append(one_mat)
                        if debug:
                            phi = att_mat_dict[at] @ one_mat
                            phi_Total = phi @ phi.T
                            phi_list.append(phi_Total)
                            ones = np.ones([self.domains[at],1])
                            numerator = np.linalg.norm(att_mat_dict[at] @ ones, 2) ** 2
                            denominator = self.domains[at] ** 2
                            product_Abar = product_Abar * numerator / denominator
                            identity_product_Abar = identity_product_Abar * 1/self.domains[at]**2
                recon_answer = res_mech.get_recon_answer(mat_list)
                noisy_answer_vector += recon_answer
                if debug:
                    csv_data.append(res_mech.get_csv_data())
                    recon_true = res_mech.get_origin_answer(mat_list)
                    non_noisy_vector += recon_true
                    # Kronecker product of phi_list
                    if len(phi_list) > 0:
                        phi_result = phi_list[0]
                        for j in range(1, len(phi_list)):
                            phi_result = np.kron(phi_result, phi_list[j])
                    else:
                        phi_result = None
                    # 7.1
                    covar_result.append(res_mech.noise_level*phi_result if phi_result is not None else None)
                    # 7.2
                    trace_result.append(res_mech.noise_level*product_A*product_Abar)
                    # 4.1
                    identity_cov_result.append(res_mech.noise_level*identity_product_A*identity_product_Abar)
                    # 4.2
                    identity_trace_result.append(res_mech.noise_level*identity_product_B*identity_product_Abar)
            if len(att_mat_list) > 0:
                pass # Optionally apply kron_result to noisy/non_noisy_vector if needed
            mech.input_noisy_answer_vector(noisy_answer_vector)
            if debug:
                mech.input_non_noisy_vector(non_noisy_vector)
                mech.input_csv_data(csv_data)
                mech.input_covar_result(sum([x for x in covar_result if x is not None]))
                mech.input_trace_result(sum(trace_result))
                mech.input_identity_cov_result(sum(identity_cov_result))
                mech.input_identity_trace_result(sum(identity_trace_result))
                debug_results[att] = {
                    'csv_data': csv_data,
                    'non_noisy_vector': non_noisy_vector,
                    'covar_result': covar_result,
                    'trace_result': trace_result,
                    'identity_cov_result': identity_cov_result,
                    'identity_trace_result': identity_trace_result,
                }
        if debug:
            return debug_results

    def get_mean_error(self, ord=1):
        error_list = []
        N = len(self.data)
        for att in self.marg_dict:
            mech = self.marg_dict[att]
            noisy_answer_vector = mech.get_noisy_answer_vector()
            non_noisy_vector = mech.get_non_noisy_vector()
            csv_data = mech.get_csv_data()
            #error2=np.linalg.norm(non_noisy_vector-csv_data,ord=ord)
            l_error = np.linalg.norm(noisy_answer_vector - non_noisy_vector, ord=ord)
            error_list.append(l_error / N)
            #error_list.append(error2 / N)
        mean_error = np.mean(error_list)
        return mean_error


class ResPlanSum(ResidualPlanner):

    def __init__(self, domains, bases=None, subtract_version='v1'):
        super().__init__(domains, bases, subtract_version=subtract_version)
        self.preprocessing()

    def preprocessing(self):
        from .utils import subtract_matrix, subtract_matrix_v2
        for i, k in enumerate(self.domains):
            base = self.bases[i]
            Bs, R, Us, Ur, gamma = find_residual_basis_sum(k, base)
            if base == 'I':
                if self.subtract_version == 'v2':
                    self.residual_matrix[k] = subtract_matrix_v2(k)
                    self.gamma_matrix[k] = subtract_matrix_v2(k)
                else:
                    self.residual_matrix[k] = subtract_matrix(k)
                    self.gamma_matrix[k] = subtract_matrix(k)
            else:
                self.residual_matrix[k] = R
                self.gamma_matrix[k] = gamma
            self.residual_pinv[k] = np.linalg.pinv(self.residual_matrix[k])
            self.one_mat[k] = np.ones([k, 1]) / k

            self.pcost_sum[i] = 1
            self.var_sum[i] = np.trace(Us @ Us.T)
            # Unified Theorem 7 & 8 formulas
            R_cur = self.residual_matrix[k]
            G_cur = self.gamma_matrix[k]
            GGT_inv = np.linalg.inv(G_cur @ G_cur.T)
            self.pcost_res[i] = np.max(np.diag(R_cur.T @ GGT_inv @ R_cur))
            self.var_res[i] = np.linalg.norm(Ur @ G_cur, 'fro') ** 2

    def input_mech(self, att):
        """Store coefficients for variance and privacy cost."""
        # E.G att=(0,1) this one would calculate the variance of (0,1)
        #TODO find where it stored and how to retival it from reconstruction.

        # att = (2, 3) means workload on attributes 2 and 3
        mech = MargMech(self.domains, att)
        self.marg_dict[att] = mech
        self.marg_index[att] = self.num_of_mech
        self.num_of_mech += 1

        # the subsets of (2, 3) are (,), (2,), (3,) and (2, 3)
        # these are residual basis needed to reconstruct workload on attribute (2, 3)
        att_subsets = all_subsets(att)
        for subset in att_subsets:
            if subset not in self.res_dict:
                pcost_res_list = [self.pcost_res[at] for at in subset]
                pcost_coeff = np.prod(pcost_res_list)
                # save the privacy cost coefficient for the residual mechanism
                self.pcost_coeff[subset] = pcost_coeff

                #TODO: add one extra parameters to use identity matrix or not
                res_mech = ResMech(self.domains, subset, self.residual_matrix, self.gamma_matrix)
                self.res_dict[subset] = res_mech
                self.res_index[subset] = self.num_of_res
                self.id2res[self.num_of_res] = subset
                self.num_of_res += 1

        for subset in att_subsets:
            var_res_list = [self.var_res[at] for at in subset]
            # trace (A kron B) = trace(A) * trace(B)
            var_res_query = np.prod(var_res_list)
            var_sum_list = []
            for c in att:
                if c not in subset:
                    var_sum_list.append(self.var_sum[c])
            var_sum_query = np.prod(var_sum_list)
            self.var_coeff_sum[subset] += var_sum_query * var_res_query

            # check numeric overflow
            if var_sum_query * var_res_query < 0:
                print(subset, var_sum_query, var_res_query)
                print("numeric overflow! var<0: ", var_sum_query * var_res_query)

    def output_coefficient(self):
        pcost_coeff_list = []
        var_sum_list = []
        for res in self.res_index.keys():
            pcost_coeff_list.append(self.pcost_coeff[res])
            var_sum_list.append(self.var_coeff_sum[res])
        v = np.array(var_sum_list)
        p = np.array(pcost_coeff_list)
        return v, p

    def get_noise_level(self):
        # Calculate coefficients and noise levels normally
        param_v, param_p = self.output_coefficient()
        x_sum, obj = find_var_sum_cauchy(param_v, param_p, c=1.0)
        if x_sum is None:
            print("failed to find a solution")
            return 0
        for i, noise_level in enumerate(x_sum):
            att = self.id2res[i]
            res_mech = self.res_dict[att]
            res_mech.input_noise_level(noise_level)
        return obj

    def get_random_noise_level(self, seed=None):
        # Skip coefficient calculation and directly add random noise
        if seed is not None:
            np.random.seed(seed)
        for i in range(len(self.res_index.keys())):
            att = self.id2res[i]
            res_mech = self.res_dict[att]
            # Input positive random noise to avoid sqrt issues
            res_mech.input_noise_level(abs(np.random.normal(1, 0.5)))  # Use abs to ensure positive values
        return 0

    def get_zero_noise_level(self):
        # Calculate coefficients but set all noise levels to 0

        for i in range(len(self.res_index.keys())):
            att = self.id2res[i]
            res_mech = self.res_dict[att]
            res_mech.input_noise_level(0)
        return 0


class ResPlanMax(ResidualPlanner):

    def __init__(self, domains, bases=None):
        super().__init__(domains, bases)
        self.num_of_query = 0
        self.preprocessing()

    def preprocessing(self):
        for i, k in enumerate(self.domains):
            base = self.bases[i]
            Bs, R, Us, Ur = find_residual_basis_max(k, base)
            self.residual_matrix[k] = R
            self.residual_pinv[k] = np.linalg.pinv(R)
            self.one_mat[k] = np.ones([k, 1]) / k

            self.pcost_sum[i] = 1
            self.pcost_res[i] = np.max(np.diag(R.T @ np.linalg.inv(R @ R.T) @ R))
            self.var_sum[i] = np.diag(Us @ Us.T)
            self.var_res[i] = np.diag(Ur @ Ur.T)

    def input_mech(self, att, var_bound=1.0):
        """Get variance for each query."""

        mech = MargMech(self.domains, att)
        self.marg_dict[att] = mech
        self.marg_index[att] = self.num_of_mech
        self.num_of_mech += 1

        # the subsets of (2, 3) are (,), (2,), (3,) and (2, 3)
        # these are residual basis needed to reconstruct workload on attribute (2, 3)
        att_subsets = all_subsets(att)
        for subset in att_subsets:
            if subset not in self.res_dict:
                pcost_res_list = [self.pcost_res[at] for at in subset]
                pcost_coeff = np.prod(pcost_res_list)
                # save the privacy cost coefficient for the residual mechanism
                self.pcost_coeff[subset] = pcost_coeff

                res_mech = ResMech(self.domains, subset, self.residual_matrix)
                self.res_dict[subset] = res_mech
                self.res_index[subset] = self.num_of_res
                self.id2res[self.num_of_res] = subset
                self.num_of_res += 1

        row_list = []
        col_list = []
        var_list = []
        # cur_id = self.marg_index[att]
        #TODO: for range query the row size should change
        num_of_rows = np.prod([self.domains[c] for c in att])

        for subset in att_subsets:
            var_coeff_list = []
            for c in att:
                if c in subset:
                    var_coeff_list.append(self.var_res[c])
                if c not in subset:
                    var_coeff_list.append(self.var_sum[c])
            if len(var_coeff_list) == 0:
                var_coeff = 1.0
            else:
                var_coeff = reduce(lambda xx, yy: np.kron(xx, yy), var_coeff_list)

            # check the size
            if len(var_coeff_list) == 0:
                row_size = 1
            else:
                row_size = len(var_coeff)
            assert row_size == num_of_rows
            if row_size > 10**6:
                print(subset)
                print("number of rows too large!", row_size)

            if len(var_coeff_list) == 0:
                var_list.append(var_coeff)
            else:
                var_list.extend(var_coeff)
            row_ids = self.num_of_query + np.arange(row_size)
            row_list.extend(row_ids)
            sub_id = self.res_index[subset]
            col_list.extend([sub_id] * row_size)

        self.sparse_row[att] = np.array(row_list)
        self.sparse_col[att] = np.array(col_list)
        self.var_coeff[att] = np.array(var_list)
        self.var_bound[att] = np.array([var_bound] * int(num_of_rows))
        self.num_of_query += int(num_of_rows)

    def output_coeff(self):
        pcost_coeff_list = []
        var_coeff_list = []
        row_list = []
        col_list = []
        var_bound_list = []
        for res in self.res_index.keys():
            pcost_coeff_list.append(self.pcost_coeff[res])
        for mech in self.marg_index.keys():
            var_coeff_list.append(self.var_coeff[mech])
            row_list.append(self.sparse_row[mech])
            col_list.append(self.sparse_col[mech])
            var_bound_list.append(self.var_bound[mech])
        coeff = np.array(pcost_coeff_list)
        val = np.concatenate(var_coeff_list)
        row = np.concatenate(row_list)
        col = np.concatenate(col_list)
        A = sp.csr_matrix((val, (row, col)), shape=(
            self.num_of_query, self.num_of_res))
        b = np.concatenate(var_bound_list)
        return coeff, A, b

    def get_noise_level(self):
        coeff, A, b = self.output_coeff()
        var, obj = find_var_max_cvxpy(coeff, A, b)
        #var, obj = find_var_max_gurobi(coeff, A, b)
        for i, noise_level in enumerate(var):
            att = self.id2res[i]
            res_mech = self.res_dict[att]
            res_mech.input_noise_level(noise_level)
        return obj


def test_prefixsum():
    domains = [16]
    bases = ['P']
    system = ResPlanSum(domains, bases)
    att = tuple(range(len(domains)))
    total = 0
    for i in range(1, 2):
        subset_i = list(itertools.combinations(att, i))
        print("num of marginals: ", len(subset_i))
        for subset in subset_i:
            cur_domains = [domains[c]+0.0 for c in subset]
            num_query = np.product(cur_domains)
            system.input_mech(subset)
            total += num_query
    print("total num of queries", total)
    return system, total


def test_prefixsum_max():
    domains = [16]
    bases = ['P']
    system = ResPlanMax(domains, bases)
    att = tuple(range(len(domains)))
    total = 0
    for i in range(1, 2):
        subset_i = list(itertools.combinations(att, i))
        print("num of marginals: ", len(subset_i))
        for subset in subset_i:
            cur_domains = [domains[c]+0.0 for c in subset]
            num_query = np.product(cur_domains)
            system.input_mech(subset)
            total += num_query
    print("total num of queries", total)
    return system, total


def test_Adult():
    domains = [85, 9, 100, 16, 7, 15, 6, 5, 2, 100, 100, 99, 42, 2]
    col_names = ['age', 'workclass', 'fnlwgt', 'education-num', 'marital-status',
       'occupation', 'relationship', 'race', 'sex', 'capital-gain',
       'capital-loss', 'hours-per-week', 'native-country', 'income>50K']
    bases = ['P', 'I', 'P', 'I', 'I', 'I', 'I', 'I', 'I', 'P', 'P', 'P', 'I', 'I']
    system = ResPlanSum(domains, bases)

    data = pd.read_csv(os.path.join(_DATA_DIR, "adult.csv"))
    system.input_data(data, col_names)
    print("Len of adult dataset: ", len(data))

    att = tuple(range(len(domains)))
    total = 0
    for i in range(1, 2):
        subset_i = list(itertools.combinations(att, i))
        print("Num of " + str(i) + "-way marginals: ", len(subset_i))
        for subset in subset_i:
            system.input_mech(subset)
            cur_domains = [domains[c] for c in subset]
            total += np.multiply.reduce(cur_domains)
    print("Total num of queries: ", total, "\n")
    return system, total


def test_simple_adult():
    # Define domains for education(3), marital(2), gender(2)
    domains = [2, 2, 3]
    col_names = ['education', 'marital', 'gender']
    # Using Identity basis for all columns since they are small domains
    bases = ['I', 'I', 'I']
    system = ResPlanSum(domains, bases)

    data = pd.read_csv(os.path.join(_DATA_DIR, "simple_adult.csv"))
    system.input_data(data, col_names)
    print("Len of simple adult dataset: ", len(data))

    att = tuple(range(len(domains)))
    total = 0
    for i in range(2, 3):
        subset_i = list(itertools.combinations(att, i))
        print("Num of " + str(i) + "-way marginals: ", len(subset_i))
        for subset in subset_i:
            system.input_mech(subset)
            cur_domains = [domains[c] for c in subset]
            total += np.multiply.reduce(cur_domains)
    print("Total num of queries: ", total, "\n")
    return system, total


def test_paper():
    # Define domains for gender(2) and income(3) as shown in the range queries example
    domains = [2, 3]  # Gender: Male/Female, Income: Low/Med/High
    col_names = ['gender', 'income']
    # Using Range basis for both attributes to enable range queries
    bases = ['I', 'R']
    system = ResPlanSum(domains, bases)

    # Create synthetic data matching the example: x = (200, 300, 100, 220, 260, 80)
    # This represents: Male-Low=200, Male-Med=300, Male-High=100, Female-Low=220, Female-Med=260, Female-High=80
    data_list = []
    counts = [200, 300, 100, 220, 260, 80]  # x0, x1, x2, x3, x4, x5 from the example
    
    # Generate individual records based on the counts
    # Gender: 0=Male, 1=Female; Income: 0=Low, 1=Med, 2=High
    combinations = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]
    
    for i, (gender, income) in enumerate(combinations):
        for _ in range(counts[i]):
            data_list.append([gender, income])
    
    data = pd.DataFrame(data_list, columns=col_names)
    system.input_data(data, col_names)
    print("Len of range queries example dataset: ", len(data))
    print("Data distribution:")
    print("Male-Low:", counts[0], "Male-Med:", counts[1], "Male-High:", counts[2])
    print("Female-Low:", counts[3], "Female-Med:", counts[4], "Female-High:", counts[5])

    # Add range queries for the example
    # Att1: Male with low-med income -> attributes (0,) with range query
    # Att2: Medium-High income -> attributes (1,) with range query  
    # Also add the 2-way marginal for the full table
    
    att = tuple(range(len(domains)))
    total = 0
    
    # Add 1-way marginals (for individual range queries)
    for i in range(1, 2):
        subset_i = list(itertools.combinations(att, i))
        print("Num of " + str(i) + "-way marginals: ", len(subset_i))
        for subset in subset_i:
            system.input_mech(subset)
            cur_domains = [domains[c] for c in subset]
            total += np.multiply.reduce(cur_domains)
            
    print("Total num of queries: ", total, "\n")
    return system, total



def test_age_income():
    # Define domains for age(85) and income(2)
    domains = [85, 2]
    col_names = ['age', 'income']
    # Using Prefix basis for age (large domain) and Identity basis for income (small domain)
    bases = ['P', 'P']
    system = ResPlanSum(domains, bases)

    data = pd.read_csv(os.path.join(_DATA_DIR, "age_income.csv"))
    system.input_data(data, col_names)
    print("Len of age_income dataset: ", len(data))

    att = tuple(range(len(domains)))
    total = 0
    for i in range(1, 2):
        subset_i = list(itertools.combinations(att, i))
        print("Num of " + str(i) + "-way marginals: ", len(subset_i))
        for subset in subset_i:
            system.input_mech(subset)
            cur_domains = [domains[c] for c in subset]
            total += np.multiply.reduce(cur_domains)
    print("Total num of queries: ", total, "\n")
    return system, total


def test_allkway_csv(n, d=5, k=3):
    domains = [n] * d
    col_names = [str(i) for i in range(d)]
    bases = ['P'] * d
    system = ResPlanMax(domains, bases)
    data_numpy = np.zeros([10_000, d])
    df = pd.DataFrame(data_numpy, columns=col_names)
    system.input_data(df, col_names)

    att = tuple(range(len(domains)))
    total = 0
    for i in range(0, k+1):
        subset_i = list(itertools.combinations(att, i))
        print("Num of " + str(i) + "-way marginals: ", len(subset_i))
        for t, subset in enumerate(subset_i):
            system.input_mech(subset)
            cur_domains = [domains[c] for c in subset]
            total += np.multiply.reduce(cur_domains)
            if t % 10_000 == 0 and t > 0:
                print("Selecting marginal: ", t)
    # print("total num of queries: ", total, "\n")
    return system, total


if __name__ == '__main__':
    start = time.time()

    pcost = 1
    #system, total = test_Adult()
    #system, total = test_simple_adult()
    #system, total = test_age_income()  # Using the new age_income test
    system, total = test_paper()  # Using the range queries example
    obj = system.get_noise_level()
    print("obj: ", obj)
    system.measurement()
    system.reconstruction()
    l_error = system.get_mean_error(ord=1)
    print("Mean Error: ", l_error)

    end = time.time()
    print("time is: ", end-start)

