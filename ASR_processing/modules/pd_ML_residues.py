import pandas as pd
from modules.pd_post_dist import pd_post_dist
from modules.read_rst import read_rst

def pd_ML_residues(infile, nodes, seq_length): 
    for i, node in enumerate(nodes):
        rst = read_rst(infile, node_point=int(node), seq_length=seq_length)
        df = pd_post_dist(rst)
        df = df.astype(float)
        df[20] = df.idxmax(axis=1)
        df.columns = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K',
                      'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V','res_idx']
        idx_to_res = {}
        for n,q in enumerate(df.columns):
            if q == 'res_idx':
                continue
            else:
                idx_to_res[n] = q
        df['res'] = df['res_idx'].map(idx_to_res)
        ML_res = df['res'].to_list()
        df = df.drop(['res_idx', 'res'], axis=1)
        df['ML'] = df.max(axis =1)
        ML_prob = []
        ML_prob = df['ML'].to_list()
        if i == 0:
            ml_df = pd.DataFrame(ML_prob).transpose()
            res_df = pd.DataFrame(ML_res).transpose()
        else:
            ml_df = pd.concat([ml_df, pd.DataFrame(ML_prob).transpose()], axis = 0)
            res_df = pd.concat([res_df, pd.DataFrame(ML_res).transpose()], axis = 0)
    ml_df.columns = list(range(1, seq_length+1))
    res_df.columns = list(range(1, seq_length+1))
    ml_df.index = nodes
    res_df.index = nodes
    return [ml_df, res_df]