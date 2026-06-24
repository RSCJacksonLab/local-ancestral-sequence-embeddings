import pandas as pd
import numpy as np

def branch_mutations(infile, no_branches):
    with open(infile, 'r') as file:
        branches_out = {}
        current_branch = None
        for line in file.readlines():
            if line.startswith("Branch "):
                current_branch = int(line.split()[1][:-1])
                branch_list = branches_out.setdefault(current_branch, [])
            elif line.strip() == "List of extant and reconstructed sequences":
                break
            elif line.strip() and current_branch:
                branch_list.append(line.split())
    new_dict = {}
    for key, value in branches_out.items():
        new_dict[key] = pd.DataFrame(value) 
    branch = []
    muts_list = []       
    for x in range(1, (no_branches+1)):
        z = []
        df = new_dict[x]
        if df.empty == True:
            continue
        else:
            df = new_dict[x]
            df[4] = df[4].replace({'-':np.nan})
            df = df.dropna()
            pos = df[0].tolist()
            ori = df[1].tolist()
            mut = df[4].tolist()
            for p,o,m in zip(pos, ori, mut):
                comb = str(o)+str(p)+str(m)
                z.append(comb) 
                y = '/'.join(z)
            branch.append(x)
            muts_list.append(y)
    branch_muts_df = pd.DataFrame(branch)
    branch_muts_df[1] = muts_list
    branch_muts_df.columns = ['branch', 'mutations']
    return new_dict, branch_muts_df