import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

def format_dummy_col(feat_col, dummy_col):
    """Handles column names for dummy data
    
    Args:
        feat_col: Name of the column with the input feature
        dummy_col: String of the dummy column 
        
    Returns:
        Dummy column with better formatting
    """
    out = dummy_col.replace("(", "")\
                   .replace("]", "")\
                   .replace(".0", "")\
                   .replace(", ", "|")
    
    return feat_col + '_' + out


def setup_numtree_prepoc(df, num_features, col_exposure, col_target, max_depth=4, min_impurity_decrease=0.1e-6):
    """Sets up a numeric preprocessing with tree based discretization
    
    Args:
        df: Pandas DataFrame with the input data
        num_features: List with the names of numeric features
        col_exposure: String exposure column name
        max_depth: Int tree parameter
        min_impurity_decrease: Float ree parameter
        
    Returns:
        Dictionary, of dictionaries with the following contents:
        {'feature1': {'thresholds': list with the tree based cuts}
         'feature2': ...}
    """
    # Dictionary where we will store cuts for each feature
    dict_enc_num = {feat: {} for feat in num_features}
    for num_feature in num_features:
        tree_model = DecisionTreeRegressor(max_depth=max_depth, min_impurity_decrease=min_impurity_decrease)
        tree_model.fit(df[num_feature].values.reshape(-1, 1), df[col_target], sample_weight=df[col_exposure].values)
        preds = tree_model.predict(df[num_feature].values.reshape(-1, 1))
        # Extract the tresholds from the tree
        thresholds = sorted(list(set(tree_model.tree_.threshold.tolist())))
        # The first value is filled with -2
        # which is the default th for a leaf node
        thresholds = thresholds[1:]
        thresholds.insert(0, -np.Inf)
        thresholds.append(np.Inf)
        dict_enc_num[num_feature]['thresholds'] = thresholds
    return dict_enc_num

def apply_treepreproc(dict_enc, df):
    """Applies a bucketing based preprocessing with 
    tree based discretization to a dataframe
    
    Args:
        dict_enc: Dictionary with the preprocessing parameters
        df: Pandas DataFrame with the input data
        
    Returns:
        horiz_concat: numpy.ndarray with the preprocessed data
        lst_col_names: all bucketed column names
    """
    lst_mat_enc = []
    lst_col_names = []
    #lst_col_names_drop = []
    # for all feature apply encoding
    for feat in dict_enc.keys():
        #enc = dict_enc[feat]['enc']
        # drop the first array column so there is 
        # no mutual information issues
        thresholds = dict_enc[feat]['thresholds']
        df[feat + '_tree'] = pd.cut(df[feat], thresholds, include_lowest=True)
        dumm_df = pd.get_dummies(df[feat + '_tree'], drop_first=True)
        dumm_df.columns = [format_dummy_col(feat, str(c)) for c in dumm_df.columns]
        lst_mat_enc.append(dumm_df)
        lst_col_names = lst_col_names + dumm_df.columns.tolist()
        #lst_col_names = lst_col_names + dict_enc[feat]['col_names']
        #lst_col_names_drop = lst_col_names_drop + dict_enc[feat]['col_names_drop']
    # horizontal concat
    horiz_concat = np.concatenate(lst_mat_enc, axis=1)
    return horiz_concat, lst_col_names

def setup_numeric_prepoc(df, num_features, n_bins, strategy):
    """Sets up the numeric preprocessing. It is based in sklearn.preprocessing KBinsDiscretizer
    
    Args:
        df: Pandas DataFrame with the input data
        num_features: List with the names of numeric features
        n_bins: Number of bins to be generated
        strategy: klearn.preprocessing KBinsDiscretizer strategy: ['uniform', 'quantile', 'kmeans']
        
    Returns:
        Dictionary, of dictionaries with the following contents:
        {'feature1': {'enc': fitted KBinsDiscretizer,
                      'col_names': default bucketed column names
                      'col_names_drop': bucketed column names, after removing the 
                                        first bucket}
         'feature2': ...}
    """
    # Dictionary where we will store cuts for each feature
    dict_enc_num = {feat: {} for feat in num_features}

    for num_feature in num_features:
        # Transform the dataset with KBinsDiscretizer
        # using a kmeans strategy
        enc = KBinsDiscretizer(n_bins=n_bins, encode='onehot-dense', strategy=strategy)
        enc.fit(df[num_feature].values.reshape(-1, 1))
        # capture data edges
        edges = enc.bin_edges_[0].tolist()
        edges_name_lst = []
        for i in range(len(edges) - 1):
            start = str(round(edges[i], 1))
            end = str(round(edges[i + 1], 1))
            suff = '{}_{}'.format(start, end)
            edges_name_lst.append(suff)
        #capture all column names
        col_names = ['{}_{}'.format(num_feature, suff) for suff in edges_name_lst]
        #capture all but the first column names
        col_names_drop = col_names[1:]
        # save in encoding dictionary
        dict_enc_num[num_feature]['enc'] = enc
        dict_enc_num[num_feature]['col_names'] = col_names
        dict_enc_num[num_feature]['col_names_drop'] = col_names_drop
    return dict_enc_num

def get_lowsupp_column(df, feature, other_values, fill_value='Other'):
    df[feature + '_other'] = df[feature]
    df.loc[df[feature + '_other'].isin(other_values), feature + '_other'] = fill_value
    
def setup_categoric_prepoc(df, cat_features, low_supp_TH):
    """Sets up the categoric preprocessing. It is based in sklearn.preprocessing OneHotEncoder
    
    Args:
        df: Pandas DataFrame with the input data
        cat_features: List with the names of categoric features
        supp_TH: Threshold used, which will convert combine low support categories to 'Other'
        
    Returns:
        Dictionary, of dictionaries with the following contents:
        {'feature1': {'enc': fitted OneHotEncoder,
                      'col_names': default bucketed column names
                      'col_names_drop': bucketed column names, after removing the 
                                        first bucket}
         'feature2': ...}
    """
    dict_enc_cat = {feat: {} for feat in cat_features}
    # capture low support entries
    for cat_feature in cat_features:
        supp_series = 100. * df[cat_feature].value_counts() / len(df)
        other_values = supp_series[supp_series < low_supp_TH].index.values.tolist()
        dict_enc_cat[cat_feature]['others'] = other_values
        get_lowsupp_column(df, cat_feature, other_values)
        # learn one hot encoder
        enc = OneHotEncoder(sparse=False).fit(df[cat_feature + '_other'].values.reshape(-1, 1))
        # capture data categories, keep first 5 characters
        categories = [c[:6] for c in enc.categories_[0].tolist()]
        # capture final col names
        col_names = ['{}_{}'.format(cat_feature, c) for c in categories]
        #capture all but the first column name
        col_names_drop = col_names[1:]
        # save in encoding dictionary
        dict_enc_cat[cat_feature]['enc'] = enc
        dict_enc_cat[cat_feature]['col_names'] = col_names
        dict_enc_cat[cat_feature]['col_names_drop'] = col_names_drop
    return dict_enc_cat

def apply_encoding(dict_enc, df):
    """Applies a bucketing based preprocessing to a dataframe. The input dictionary
    should be returned by setup_numeric_prepoc or setup_categoric_prepoc
    
    Args:
        dict_enc: Dictionary with the preprocessing parameters
        df: Pandas DataFrame with the input data
        
    Returns:
        horiz_concat: numpy.ndarray with the preprocessed data
        lst_col_names: all bucketed default column names
        lst_col_names_drop: actual bucketed column names after dropping the 
                            bucket per feature
    """
    lst_mat_enc = []
    lst_col_names = []
    lst_col_names_drop = []
    # for all feature apply encoding
    for feat in dict_enc.keys():
        # if it is a categorical encoding
        if 'others' in dict_enc[feat].keys():
            feat_oth = feat + '_other'
            # check if the '_other' version column DOES NOT exists
            if feat_oth not in df.columns:
                # generate the '_other' column version
                get_lowsupp_column(df, feat, dict_enc[feat]['others'], fill_value='Other')
        else:
            feat_oth = feat 
        enc = dict_enc[feat]['enc']
        # drop the first array column so there is 
        # no mutual information issues
        lst_mat_enc.append(enc.transform(df[feat_oth].values.reshape(-1, 1))[:, 1:])
        lst_col_names = lst_col_names + dict_enc[feat]['col_names']
        lst_col_names_drop = lst_col_names_drop + dict_enc[feat]['col_names_drop']
    # horizontal concat
    horiz_concat = np.concatenate(lst_mat_enc, axis=1)
    return horiz_concat, lst_col_names, lst_col_names_drop


#visualization functions
def get_cut_df(feat_col, input_slider, n_bins, df, target_col):
    """
    Handles the type of the data to generate the intermediate datadframe
    """
    if df[feat_col].dtype in [int, float, np.number]:
        return df_vol_br_num(feat_col, input_slider, n_bins, df, target_col)
    else:
        return df_vol_br_cat(feat_col, input_slider, n_bins, df, target_col)

#capture volume / BR df for numerical variables
def df_vol_br_num(feat_col, input_slider, n_bins, df, obj_col):
    """
    Generate the intermediate dataframe with number of observations and 
    number of bads per bin. Specific for numerical features.
    """
    #get the numeric input from the dual slider
    perc_sliders = [v/100. for v in input_slider]
    var_lims = df[feat_col].quantile([perc_sliders[0], perc_sliders[1]]).values
    v_min, v_max = var_lims[0], var_lims[1]
    #filter the dataset using the slider input
    df_cut = df.loc[(df[feat_col] <= v_max) & (df[feat_col] >= v_min)][[obj_col, feat_col]]
    #number of cuts = minumum of n_bins, number of unique values of the variable
    n_cuts = min(int(n_bins), df_cut[feat_col].nunique())
    cuts = [c for c in np.linspace(v_min, v_max, n_cuts + 1)]
    if cuts[-1] < v_max:
        cuts.append(v_max)
    cut_col = feat_col + '_'
    df_cut[cut_col] = pd.cut(df_cut[feat_col], cuts, include_lowest=True)
    #generate aggregated values
    N = df_cut.groupby(cut_col)[feat_col].count().values
    TR = df_cut.groupby(cut_col)[obj_col].mean().values
    cuts = df_cut.groupby(cut_col)[feat_col].count().index.astype(str).values
    #handle NA entries
    if df[feat_col].isna().sum() > 0:
        N = np.append(([df[feat_col].isna().sum()]), N)
        TR = np.append(([df.loc[df[feat_col].isna()][obj_col].mean()]), TR)
        cuts =  np.append(['NA'], cuts)
    #generate global transformation rate
    return (pd.DataFrame({'cuts': cuts,
                         'N': N,
                         'BR': TR}), df_cut[obj_col].mean())

#capture volume / BR df for categorical variables
def df_vol_br_cat(feat_col, input_slider, n_bins, df, target_col):
    """
    Generate the intermediate dataframe with number of observations and 
    number of bads per bin. Specific for categorical features.
    """
    #pick top n_bins levels by volume
    cut_levels = df.groupby(feat_col)[feat_col].count().sort_values(ascending=False)[:int(n_bins)].index.values.tolist()
    df_cut = df.loc[df[feat_col].isin(cut_levels)]
    #capture volumes
    N = df_cut.groupby(feat_col)[feat_col].count().values
    #capture transformations
    TR = df_cut.groupby(feat_col)[target_col].mean().values
    return (pd.DataFrame({'cuts': df_cut.groupby(feat_col)[feat_col].count().index.astype(str).values,
                         'N': N,
                         'BR': TR}), df_cut[target_col].mean())


def output_graph_update(feat_col, input_slider, n_bins, df, obj_col):
    """
    Generate the plotly plot showing the visualization of the intermediate 
    dataframe with volume and bad rate per bin.
    """
    #get the df with volume and bad rate
    df_tr, avg_tr = get_cut_df(feat_col, input_slider, n_bins, df, obj_col)
    #line represents transformation rate
    tr_line = go.Scatter(x = df_tr.cuts,
                         y = df_tr.BR,
                         yaxis = 'y2',
                         name = obj_col)
    #bar represents volume @ cut
    vol_bars = go.Bar(x = df_tr.cuts,
                      y = df_tr.N,
                      name = 'Volume')
    #avg line
    avg_line = go.Scatter(x = df_tr.cuts,
                          y = np.repeat(avg_tr, df_tr.shape[0]),
                          yaxis = 'y2',
                          name = 'AVG {}'.format(obj_col),
                          line = dict(
                              color = ('rgb(205, 0, 0)')
                                     )
                         )
    #small layout
    layout = go.Layout(
            title = '{} for {}'.format(obj_col, feat_col),
            yaxis = dict(title = 'Volume',
                         range = [0, max(df_tr.N)]),
            yaxis2 = dict(title = obj_col,
                         overlaying='y',
                         side='right',
                         range = [0, max(df_tr.BR) + 0.05*max(df_tr.BR)])

        )
    return {'data': [vol_bars, tr_line, avg_line],
            'layout': layout}