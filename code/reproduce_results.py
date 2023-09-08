
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_variable_significance import *

Ms = np.arange(101,201,1) # Monte Carlo repetition ids




def get_name_of_all_covariates(Pc):
    '''
    get the names of all covariates

    Parameter
    ---------
    Pc: int,
        number of characteristics, and the number of covariates = 2*Pc
    
    Return
    ------
    covar_names: list,
        names of all covariates
    '''
    covar_names = []
    for covar_idx in range(0,Pc):
        covar_names.append('c'+str(covar_idx+1))
    for covar_idx in range(Pc,2*Pc):
        covar_names.append('c'+str(covar_idx-Pc+1)+'mt')
    return covar_names

def get_covariate_name_in_math_text_from_idx(Pc,covar_idx): 
    '''
    get the name of a covariate in math text

    Parameter
    ---------
    Pc: int,
        number of characteristics
    covar_idx: int, 
        index of a covariate in the covariate list.

    Return
    ------
    covariate_math_format: str
        covariate name in math format
    '''
    if covar_idx<Pc:
        covariate_math_format = r'$c_{i,' + str(covar_idx+1) + ',t}$'
    else:
        covariate_math_format= r'$c_{i,' +str(covar_idx-Pc+1) + ',t} \\times m_t$'
    return covariate_math_format


def load_scaled_tau_samples_MC(Pc,N,T, m, sampled_NN, tot_iter):
    '''
    load scaled tau[h_max] samples for all covariates for all different Monte Carlo repetitions.

    Parameter
    ---------
    Pc: int,
        number of characteristics
    N: int,
        number of firms
    T: int,
        number of months
    sampled_NN: int,
        number of hidden layers of sampled nueral networks in discretization
    tot_iter: int,
        number of iterations in discretization

    Return
    ------
    scaled_tau_samples_MC: ndarray, (#MC, #iters, #covars)
       scaled tau[h^(max)] samples for all covariates for all Monte Carlo repetitions
    '''

    scaled_tau_path = '../data/Pc'+str(Pc)+'/N'+str(N)+'_T'+str(T)+'/scaled_tau_samples/m'+str(m)+'_NN'+str(sampled_NN)+'/'


    scaled_tau_samples_MC = []
    for M in Ms:

        scaled_tau_samples =  np.load(scaled_tau_path+'/M'+str(M)+'/1_'+str(tot_iter)+'.npy', allow_pickle=True)
        scaled_tau_samples_MC.append(scaled_tau_samples)

    scaled_tau_samples_MC = np.array(scaled_tau_samples_MC)
    return scaled_tau_samples_MC


def load_tstat_MC(Pc, N, T, fitted_NN):
    '''
    load test statistics of all covariates for all Monte Carlo repetitions.

    Parameter
    ---------
    Pc: int,
        number of characteristics
    N: int,
        number of firms
    T: int,
        number of months
    sampled_NN: int,
        number of hidden layers of fitted nueral networks

    Return
    ------
    tstats_MC: ndarray, (num_MC,num_vars)
        test statistics of all covariates for all Monte Carlo repetitions
    '''


    tstat_path = '../data/Pc'+str(Pc)+'/N'+str(N)+'_T'+str(T)+'/test_statistics/NN'+str(fitted_NN)+'/'

    import ast
    tstats_MC = []   
    for M in Ms:
        with open(tstat_path+'/M'+str(M)+'/train_tstat.txt', 'r') as f:
            tstats = ast.literal_eval(f.read())
        tstats_MC.append(tstats) 
    tstats_MC = np.array(tstats_MC)
    
    return tstats_MC
    
 
def plot_figure2(Pc,tau_samples_MC,tstats_MC_NN_dict,scaler,covar_idx_ls):
    style = 'scientific'
    useMathText = True
    xmax = 0.018
    ymax = 23000    
    xmin = -0.0011
    xstep = 0.005
    alphas = [0.1,0.05,0.01]
    quantile_lines = ['solid','dashdotted',"dotted"]
    quantile_dotted = [(1, 1),(3, 5, 1, 5),(3, 1, 1, 1)]
    tstat_lines = ['solid','dashdot','dashed',"dotted"]
    tstat_dashes = [(10,3),(10,6),(5,1),(5,5),(5,10)]
    tstat_colors = ['m','y','g','k','c']



    i = 0
    fig, axes = plt.subplots(nrows=3,ncols=3,figsize=(20,10))
    for covar_idx in covar_idx_ls:
        covar_name = get_covariate_name_in_math_text_from_idx(Pc,covar_idx)

        tau_sample_MC = tau_samples_MC[:,:,covar_idx]
        tau_sample_MC = np.reshape(tau_sample_MC,(-1,))
        axes[int(i/3),int(i%3)].hist(tau_sample_MC,bins=100,color='b',label=r'$\hat{\mathcal{T}_j}[h^{(max)}]$')
        
        for alpha,dotted in zip(alphas,quantile_dotted):
            q = 1-alpha
            quantile_covi = np.quantile(tau_sample_MC,q=q)
            
            quantile_perc = int(q*100)
            if alpha<0.01:
                quantile_perc = round(q*100,2)
            q_label = r'$q_{'+str(quantile_perc)+r'}^{\hat{\mathcal{T}_j}[h^{(max)}]}$' 
            axes[int(i/3),int(i%3)].axvline(quantile_covi,color='r',ls='-',dashes=dotted,label=q_label)
        
        for fitted_NN,dashes,c in zip([1,2,3,4,5],tstat_dashes,tstat_colors):
            tstat_label = 'NN'+str(fitted_NN)
            tstats_MC = tstats_MC_NN_dict[tstat_label]
            tstat_MC = tstats_MC[:,covar_idx]
            scaled_tstat_MC = tstat_MC/scaler
            axes[int(i/3),int(i%3)].axvline(np.mean(scaled_tstat_MC),color=c,ls='--',dashes=dashes,label=tstat_label)
        axes[int(i/3),int(i%3)].set_title(covar_name)
        axes[int(i/3),int(i%3)].set_ylim([0,ymax])
        axes[int(i/3),int(i%3)].ticklabel_format(style=style, useMathText=useMathText,axis='y', scilimits=(0,0))
        axes[int(i/3),int(i%3)].set_xlim([xmin,xmax])
        axes[int(i/3),int(i%3)].ticklabel_format(style=style, useMathText=useMathText,axis='x', scilimits=(0,0))
        i += 1
    
    handles, labels = axes[2,1].get_legend_handles_labels()
    labels.insert(0,labels[-1])
    handles.insert(0,handles[-1])
    labels = labels[:-1]
    handles = handles[:-1]
    axes[2, 1].legend(handles=axes[2, 1].containers[0][:1],
                    bbox_to_anchor=(0.5, -0.3), loc='upper center')
    fig.tight_layout()
    axes[2, 1].legend(handles=handles,ncol=len(labels),frameon=False,
                    bbox_to_anchor=(0.5, -0.3), loc='upper center')
   



if __name__=='__main__':

    # %% Table 2
    print('reproduce results in Table 2')
    N = 200; T = 180; 
    covars_of_interest = ['c1','c2','c3mt','c1mt','c2mt','c3']
    
    columns = ['Parameter','Method']
    columns.extend(covars_of_interest)
    Pc_df = pd.DataFrame(columns=columns,dtype=object)
    Pc_df['Method'] = ['NN1','NN2','NN3','NN4','NN5']
    for panel,m,sampled_NN,tot_iter in zip(['A','B','C','D','E'],\
        [500,300,500,500,500],[1,1,1,2,3],[1000,1000,2000,1000,1000]):
        panel_df = pd.DataFrame()
        for Pc in [50,100]:
            Pc_df['Parameter'] = ['Pc='+str(Pc),'','','','']
            covar_names = get_name_of_all_covariates(Pc)
            scaled_tau_samples_MC = load_scaled_tau_samples_MC(Pc,N,T,m,sampled_NN,tot_iter)

            var_sign_freq_NN_dict = dict()
            for fitted_NN in [1,2,3,4,5]:
                print('\nPc = '+str(Pc)+' | NN'+str(fitted_NN))
                tstats_MC = load_tstat_MC(Pc, N, T, fitted_NN)
                
                var_sign_freq_NN_dict['NN'+str(fitted_NN)] = dict()
                for alpha in [0.1, 0.05, 0.01]:
                    print('variable significance frequency at the '+str(int(alpha*100))+'% level')
                    var_sign_freq_dict = variable_significance_frequency(tstats_MC,scaled_tau_samples_MC,alpha,covar_names)
                    var_sign_freq_NN_dict['NN'+str(fitted_NN)]['alpha='+str(alpha)] = var_sign_freq_dict

            for fitted_NN in [1,2,3,4,5]:
                for covar in covars_of_interest:
                    var_sign_freq_ls = []
                    for alpha in [0.1, 0.05, 0.01]:
                        var_sign_freq = var_sign_freq_NN_dict['NN'+str(fitted_NN)]['alpha='+str(alpha)][covar]
                        var_sign_freq_ls.append(int(var_sign_freq))
                    Pc_df[covar].iloc[fitted_NN-1] = var_sign_freq_ls
            panel_df = panel_df.append(Pc_df)
        panel_df.to_csv('../output/Table2Panel'+panel+'.csv',index=False)

    # %% Figure 2
    N = 200; T = 180; Pc = 100
    scaler = (int(N*T/3)**(-0.215))**2
    m = 500; sampled_NN = 1; tot_iter = 1000
        
    tstats_MC_NN_dict = dict()
    for fitted_NN in [1,2,3,4,5]:
        tstats_MC_NN_dict['NN'+str(fitted_NN)] = load_tstat_MC(Pc, N, T, fitted_NN)
    scaled_tau_samples_MC = load_scaled_tau_samples_MC(Pc,N,T,m,sampled_NN,tot_iter)
    tau_samples_MC = scaled_tau_samples_MC/scaler

    covar_names = get_name_of_all_covariates(Pc)
    covars_of_interest = ['c1','c2','c3mt','c1mt','c2mt','c3','c4','c58','c81mt']
    covar_idx_ls = [covar_names.index(v) for v in covars_of_interest]
    plot_figure2(Pc,tau_samples_MC,tstats_MC_NN_dict,scaler,covar_idx_ls)
    plt.savefig('../output/Figure2.pdf')




    # %% Figure 3
    print('reproduce results in Figure 3')
    N = 200; T = 180; 
    m = 500; sampled_NN = 1; tot_iter = 1000
    covars_of_interest = ['c1','c2','c3mt','c1mt','c2mt','c3']

    for Pc, panel in zip([50,100],['a','b']):
        covar_names = get_name_of_all_covariates(Pc)
        
        # pvals for different fitted NNs
        pvals_df = pd.DataFrame()
        for fitted_NN in [1,2,3,4,5]:
            print('\nPc = '+str(Pc)+' | NN'+str(fitted_NN))
            tstats_MC = load_tstat_MC(Pc, N, T, fitted_NN)
            scaled_tau_samples_MC = load_scaled_tau_samples_MC(Pc,N,T,m,sampled_NN,tot_iter)
            
            # pvals of interested covariates for different Monte Carlo repetitions
            pvals_MC_df = pd.DataFrame(index=Ms,columns=covars_of_interest)
            for covar in covars_of_interest:
                covar_idx = covar_names.index(covar)
                tstat_MC = tstats_MC[:,covar_idx]
                scaled_tau_sample_MC = scaled_tau_samples_MC[:,:,covar_idx]
                for M,tstat,scaled_tau_sample in zip(Ms,tstat_MC,scaled_tau_sample_MC): 
                    pval = comp_pvalue(scaled_tau_sample,tstat)
                    pvals_MC_df.loc[M,covar] = pval
            pvals_MC_df['NN'] = 'NN'+str(fitted_NN)

            pvals_df = pvals_df.append(pvals_MC_df)

        
        # box plot
        medians = pvals_df.groupby(['NN']).median()
        fig, axes = plt.subplots(2,3,figsize=(16, 8))
        for i,covar in enumerate(list(pvals_df.columns.values)[:-1]):
            ax = pvals_df.boxplot(covar, by="NN", ax=axes.flatten()[i],patch_artist=True,\
                    boxprops=dict(linestyle='-', linewidth=1.5,facecolor='b'),\
                    flierprops=dict(linestyle='-', linewidth=1.5),\
                    medianprops=dict(linestyle='-', linewidth=1.5,color='r'),\
                    whiskerprops=dict(linestyle='-', linewidth=1.5),\
                    capprops=dict(linestyle='-', linewidth=1.5),\
                    showfliers=True, grid=False, rot=0)
            ax.set(xlabel=None)
            ax.set_ylim([-0.1,1.05])
            for xtick in ax.get_xticks():
                ax.text(xtick, medians[covar].iloc[xtick-1]-0.06,"%.3f" % medians[covar].iloc[xtick-1],
                    horizontalalignment='center',size='x-small',color='r',weight='semibold')
        fig.suptitle('')    
        plt.tight_layout()
        plt.savefig('../output/Figure3('+panel+').pdf')
