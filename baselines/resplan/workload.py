import numpy as np
import itertools
from .ResPlan import ResPlanSum, ResPlanMax


def root_mean_squared_error(sum_var, num_query, pcost):
    rmse = np.sqrt(sum_var / num_query) / pcost
    return rmse


def workload_allkway(n, d, k, choice="sumvar"):
    domains = [n for _ in range(d)]
    bases = ['P'] * d
    if choice == "sumvar":
        system = ResPlanSum(domains, bases)
    elif choice == "maxvar":
        system = ResPlanMax(domains, bases)
    else:
        print("Invalid choice, choose between sumvar and maxvar.")
        return

    att = tuple(range(len(domains)))
    total = 0

    for i in range(0, k+1):
        subset_i = list(itertools.combinations(att, i))
        # print("num of marginals: ", len(subset_i))
        for subset in subset_i:
            cur_domains = [domains[c] + 0.0 for c in subset]
            num_query = np.prod(cur_domains)
            #print(subset, num_query)
            if 0 < num_query <= 5000:
                system.input_mech(subset)
                total += num_query
    return system, total


def dataset_domains(dataset):
    """Return dataset domain size for each attribute."""
    if dataset == "CPS":
        domains = [50, 100, 7, 4, 2]
        bases = ['P', 'P', 'I', 'I', 'I']
    elif dataset == "Adult":
        domains = [100, 100, 100, 99, 85, 42, 16, 15, 9, 7, 6, 5, 2, 2]
        bases = ['P'] * 5 + ['I'] * 9
    elif dataset == "Loans":
        domains = [101, 101, 101, 101, 3, 8, 36, 6, 51, 4, 5, 15]
        bases = ['P'] * 4 + ['I'] * 8
    else:
        print("Invalid Choice! Please choose between CPS, Adult and Loans")
        domains = []
        bases = []
    return domains, bases


def     workload_large_dataset(dataset, workload, choice="sumvar"):
    """Return system given workload.

    dataset: Choose between "CPS", "Adult", "Loans"
    workload: k       --> k way marginals
              "3D"    --> All 0, 1, 2, 3 way marginals
              "Small" --> All marginals with size <= 5000
    """
    domains, bases = dataset_domains(dataset)
    if choice == "sumvar":
        system = ResPlanSum(domains, bases)
    elif choice == "maxvar":
        system = ResPlanMax(domains, bases)
    else:
        print("Invalid choice, choose between sumvar and maxvar.")
        return
    num_att = len(domains)
    att = tuple(range(num_att))

    if type(workload) == int:
        lower = workload
        upper = lower + 1
    elif workload == "3D":
        lower = 0
        upper = 4
    elif workload == "Small":
        lower = 0
        upper = num_att + 1
    else:
        print("Invalid workload, choose between All, 3D, Small")
        return

    total = 0
    if choice == 'sumvar':
        for i in range(lower, upper):
            subset_i = list(itertools.combinations(att, i))
            # print("num of marginals: ", len(subset_i))
            for subset in subset_i:
                cur_domains = [domains[c] + 0.0 for c in subset]
                num_query = np.multiply.reduce(cur_domains)
                if workload == "Small":
                    if 0 < num_query <= 5000:
                        system.input_mech(subset)
                        total += num_query
                if workload == "3D" or type(workload) == int:
                    system.input_mech(subset)
                    total += num_query

    if choice == 'maxvar':
        for i in range(lower, upper):
            subset_i = list(itertools.combinations(att, i))
            # print("num of marginals: ", len(subset_i))
            for subset in subset_i:
                cur_domains = [domains[c] + 0.0 for c in subset]
                num_query = np.multiply.reduce(cur_domains)
                if 0 < num_query <= 5000:
                    system.input_mech(subset)
                    total += num_query

    return system, total
