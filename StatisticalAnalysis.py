import os
from scipy.stats import friedmanchisquare
import numpy as np
import pandas as pd
import Orange 
import matplotlib.pyplot as plt


def statistical_analyses(stat_values, names, stat_test, fig_path):
    # Performing Friedman test:
    st, pvalue = friedmanchisquare(stat_values[0,:], stat_values[1,:], stat_values[2,:], stat_values[3,:], stat_values[4,:], stat_values[5,:])
    
    # Performing post-hoc Nemenyi/Bonferroni-Dunn test:
    df = pd.DataFrame(stat_values.T)
    avg_ranks = df.rank(axis=1).mean(axis=0).values

    if stat_test == 'Nemenyi':
        # NEMENYI
        cd = Orange.evaluation.scoring.compute_CD(avg_ranks, n=stat_values.T.shape[0], alpha="0.05", test="nemenyi")
        Orange.evaluation.graph_ranks(
            avranks=avg_ranks,
            names=names,
            cd=cd,
            width=6,
            textspace=1.5,
            filename = fig_path + '-Nemenyi.eps'
        )
        # plt.show()

    else:
        # Bonferroni-Dunn:
        cd = Orange.evaluation.scoring.compute_CD(avg_ranks, n=stat_values.T.shape[0], alpha="0.05", test="bonferroni-dunn")
        Orange.evaluation.graph_ranks(
            avranks=avg_ranks,
            names=names,
            cd=cd,
            width=8,
            textspace=1.9,
            cdmethod = 0, #### REFERENCE (Bonferroni-Dunn)
            filename = fig_path + '-Bonferroni.eps'
        )
        # plt.show()

    return pvalue


def getTestSamples(PG_method:str = 'MChen', red_parameter:int = 1, metric:str = 'F1-M', classifier:str = 'MLkNN', k:int = 1, corpora:list = ['yeast']):

    out_values = list()
    names = list()

    # Load CSV file:
    df = pd.read_csv('./Results/Results_Summary.csv')

    # Extract part of the pandas datafile:
    df_excerpt = df.loc[df['NormData'] == True][df['PG_method'] == PG_method][df['Reduction_parameter'] == red_parameter][df['classifier'] == classifier][df['k'] == k][df['DDBB'].isin(corpora)][['DDBB','Exclude_imbalanced_samples','Prototype_merging',metric,'Size']]


    Prot_merging_cases = sorted(df_excerpt['Prototype_merging'].unique())
    Excluded_imb_cases = sorted(df_excerpt['Exclude_imbalanced_samples'].unique())

    for excl_imb_case in Excluded_imb_cases:
        for prot_merging_case in Prot_merging_cases:
            out_values.append(df_excerpt.loc[df_excerpt['Exclude_imbalanced_samples'] == excl_imb_case][df_excerpt['Prototype_merging'] == prot_merging_case][metric].values)
            names.append('excl{}_merg{}'.format(excl_imb_case,prot_merging_case ))
        pass
    pass

    return np.array(out_values), names, df_excerpt






if __name__ == '__main__':

    dst_path = 'StatResults'
    if not os.path.exists(dst_path): os.makedirs(dst_path)

    # Parameters:
    metric = 'F1-M'
    PG_Methods = ['MRHC', 'MChen', 'MRSP3']
    PG_Parameters = {
        'MRHC': [1],
        'MChen': [10, 50, 90],
        'MRSP3': [1]
    }
    Imbalance_cases = {
        'LowImbalance' : ['emotions', 'scene', 'birds', 'yeast', 'bibtex'],
        'HighImbalance' : ['genbase', 'medical', 'Corel5k', 'rcv1subset1', 'rcv1subset2', 'rcv1subset3', 'rcv1subset4']
    }
    Imbalance_cases = {
        'LowImbalance' : ['emotions', 'birds', 'bibtex'],
        'HighImbalance' : ['genbase', 'medical', 'Corel5k']
    }

    fout = open(os.path.join(dst_path, 'FriedmanTest.txt'), 'w')
    fout.write('Imbalance,Method,p-value\n')
    for imb_case, corpora in Imbalance_cases.items():
        dst_path_case = os.path.join(dst_path, imb_case)
        if not os.path.exists(dst_path_case): os.makedirs(dst_path_case)

        for single_PG in PG_Methods:
            for single_parameter in PG_Parameters[single_PG]:
                # Extracting values:
                stat_values, names, df_excerpt = getTestSamples(PG_method = single_PG, red_parameter = single_parameter, metric = metric, classifier = 'MLkNN', k = 1, corpora = corpora)

                # Names in a fancier manner:
                names_plot = list()
                for single_name in names:
                    temp_name = 'Exc'
                    temp_name += 'True' if 'Base' in single_name.split("_")[0] else 'False'
                    
                    if 'PolicyII' in single_name.split("_")[1]:
                        temp_name += '-MerPol2'
                    elif 'PolicyI' in single_name.split("_")[1]:
                        temp_name += '-MerPol1'
                    else:
                        temp_name += '-MerBase'

                    names_plot.append(temp_name)
                pass

                # Stat analysis:
                friedman_value = statistical_analyses(stat_values, names_plot, stat_test = 'bonferroni', fig_path = os.path.join(dst_path_case, single_PG + str(single_parameter)))
            
                #  Write Friedman score:
                fout.write("{},{},{}\n".format(imb_case,single_PG + str(single_parameter),friedman_value))
            pass
        pass
    pass
    fout.close()
    
    pass