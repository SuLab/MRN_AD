import utils
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from collections.abc import Iterable
from data_tools import plotting as gp
from data_tools import df_processing as dfp
from hetnet_ml.extractor import MatrixFormattedGraph


# TODO: Make this something that can be loaded for other network Node Types
node_color_map = {'AnatomicalEntity': "#8c88d7",
                'BiologicalProcessOrActivity': "#b75970",
                'MacromolecularMachine': "#e71761",
                'Disease': "#70c6ca",
                'Pathway': "#b1d34f",
                'PhenotypicFeature': "#154e56",
                'GeneFamily': "#5a3386",
                'PhysiologicalProcess': '#ec102f',
                'OrganismTaxon': "#e9bf98",
                'ChemicalSubstance': "#01c472",}


class MFGPathPlotter():

    def __init__(self, params='../0_data/manual/data_param.json'):

        # Read param file or store passed params
        self.params = utils.load_params(params)

        # Load required info from params file
        self.nodes, self.edges = utils.load_network(self.params)
        self.target_node, self.target_edge = utils.load_targets(self.params)
        self.hyperparam, tune_feat = utils.load_hyperparameters(self.params)
        self.model, self.features, self.coef = utils.load_model(self.params)
        self.edge_info = utils.load_semantic_info(self.params)
        self.path_stats = utils.load_path_stats(self.params)

        self._set_fresh_param_read()
        self._update_sem_info()
        self._update_graph()
        self._update_model()
        self._update_pathstats()

    def _update_sem_info(self):
        self.inv_map = self.edge_info.set_index('fwd_edge')['rev_edge'].to_dict()

    def _set_fresh_param_read(self):
        self._network_changed = False
        self._hyperparam_changed = False
        self._model_changed = False
        self._target_edge_changed = False
        self._target_node_changed = False


    def set_params(self, params):
        update_graph = self._network_changed
        update_mg = False
        update_hyperparam = self._hyperparam_changed
        update_model = self._model_changed

        new_params = utils.load_params(params)

        # Check to see what params have changed so we only make the
        # Necessary updates
        for k, v in new_params.items():
            # Don't need to change if values are the same
            if self.params.get(k, None) == v:
                continue

            # Update the network if the directory or biolink changes
            elif (k == 'nw_dir' or k == 'biolink') and not update_graph:
                update_graph = True
                update_mg = True

            elif k == 'target_node':
                update_mg = True

            elif k == 'hyperp_dir':
                old_w = self.hyperparam['w']
                self.hyperparam, tune_feat = utils.load_hyperparameters(new_params)

                # Only change if w is actually different
                # Use differnece for floating point error
                if np.abs(old_w - self.hyperparam['w']) > 1e-8:
                    update_hyperparam = True

            # Update the model if model has changed
            elif k == 'model_dir':
                update_model = True
                self.model, self.features, self.coef = utils.load_model(new_params)

            elif k == 'path_stats_dir':
                self.path_stats = utils.load_path_stats(new_params)
                self._update_pathstats()

        self.params = new_params

        # If the network changes, everything needs to be updated
        if update_graph:
            self.nodes, self.edge = utils.load_network(self.params)
            self._update_graph()
        # If the network is the same, only update the things that need it
        else:
            if update_mg:
                self.mg.set_metapaths(end_kind=self.params['target_node'])
            if update_hyperparam:
                self.mg.update_w(self.hyperparam['w'])
            if update_model:
                self._update_model()

        self._set_fresh_param_read()

    def _update_graph(self):
        self.id_to_name = self.nodes.set_index('id')['name'].to_dict()
        self.id_to_label = self.nodes.set_index('id')['label'].to_dict()
        self.node_id_to_color = self.nodes.set_index('id')['label'].map(node_color_map).to_dict()

        self.mg = MatrixFormattedGraph(self.nodes, self.edges,
                                       'ChemicalSubstance', self.params['target_node'],
                                       max_length=4, w=self.hyperparam['w'], n_jobs=30)
        self.metagraph = self.mg.metagraph

        self._update_model()


    def update_graph(self, nodes=None, edges=None, model_params=None):
        """
        Update the graph for plotting
        """

        if nodes is not None:
            self.nodes = nodes
            self._network_changed = True

        if edges is not None:
            self.edges = edges
            self._network_changed = True

        if model_params is not None:
            self.hyperparam = model_params
            self._hyperparam_changed = True

        self._update_graph()


    def _update_model(self):

        self.msat = self.model[0]
        self.max_abs = self.model[1]

        self.pos_coef = self.coef.query('coef > 0')['feature'].tolist()
        self.metapaths = self.coef.query('feature != "intercept"')['feature'].tolist()
        self.all_metapaths = self.metagraph.extract_metapaths('ChemicalSubstance', self.params['target_node'], 4)
        self.mp_info = {mp.abbrev: mp for mp in self.all_metapaths if mp.abbrev in self.metapaths}

        self.ini_means = {f: m for f, m in zip(self.metapaths, self.msat.initial_mean_)}
        self.ma_scale = {f: m for f, m in zip(self.metapaths, self.max_abs.scale_)}
        self.feat_coef = self.coef.set_index('feature')['coef'].to_dict()
        self.feat_zcoef = self.coef.set_index('feature')['zcoef'].to_dict()


    def update_model(self, model=None, coef=None):
        """
        Update the learning model used for coefficient analysis
        """
        if model is not None:
            self.model = model
        if coef is not None:
            self.coef = coef

    def _update_pathstats(self):
        self.mean_info = self.path_stats.set_index('feature')['pdp_mean'].to_dict()
        self.std_info = self.path_stats.set_index('feature')['pdp_std'].to_dict()

    def _get_model_metric(self, path_df):
        totals = path_df.groupby('metapath')['metric'].sum().to_dict()
        percent_of_total = path_df['metric'] / path_df['metapath'].map(totals)

        trans_metric = np.arcsinh(path_df['metapath'].map(totals) / path_df['metapath'].map(self.ini_means))
        scal_metric = trans_metric / path_df['metapath'].map(self.ma_scale)

        model_metric = scal_metric * path_df['metapath'].map(self.feat_coef) * percent_of_total
        return model_metric


    def _get_std_model_metric(self, path_df):

        totals = path_df.groupby('metapath')['metric'].sum().to_dict()
        percent_of_total = path_df['metric'] / path_df['metapath'].map(totals)

        trans_metric = np.arcsinh(path_df['metric'] / path_df['metapath'].map(self.ini_means))
        scal_metric = trans_metric / path_df['metapath'].map(self.ma_scale)

        model_metric = scal_metric * path_df['metapath'].map(self.feat_zcoef) * percent_of_total
        return model_metric


    def _get_scal_dwpc_metric(self, path_df):
        totals = path_df.groupby('metapath')['metric'].sum().to_dict()
        percent_of_total = path_df['metric'] / path_df['metapath'].map(totals)

        trans_metric = np.arcsinh(path_df['metapath'].map(totals) / path_df['metapath'].map(self.ini_means))
        scal_metric = trans_metric / path_df['metapath'].map(self.ma_scale)

        dwpc_metric = scal_metric * percent_of_total
        return dwpc_metric

    def _get_zscores(self, path_df):
        z_scores = []
        for row in path_df.itertuples():
            z_scores.append((row.metric - self.mean_info[row.metapath]) / self.std_info[row.metapath])
        return z_scores

    def get_path_info(self, compound, target, n_jobs=30, mps=None):
        if mps == None:
            mps = self.pos_coef

        path_df = pd.DataFrame(self.mg.extract_paths(compound, target, mps, n_jobs=n_jobs))
        if len(path_df) == 0:
            return float('nan')

        path_df['model_metric'] = self._get_model_metric(path_df)
        path_df['std_metric'] = self._get_std_model_metric(path_df)
        path_df['scal_metric'] = self._get_scal_dwpc_metric(path_df)
        path_df['z_score'] = self._get_zscores(path_df)

        return path_df

    @staticmethod
    def query_target(path_df, tgt_id):
        tgt_ids = path_df['node_ids'].apply(lambda x: x[1])

        idx = tgt_ids == tgt_id
        idx = idx[idx].index

        return path_df.loc[idx]

    @staticmethod
    def query_path_pos(path_df, tgt_id, pos=1):
        tgt_ids = path_df['node_ids'].apply(lambda x: x[pos] if len(x) > pos else float('nan'))

        if type(tgt_id) == str:
            idx = tgt_ids == tgt_id
        elif isinstance(tgt_id, Iterable):
            tgt_id = set(tgt_id)
            idx = tgt_ids.apply(lambda t: len({t} & tgt_id) > 0)
        idx = idx[idx].index

        return path_df.loc[idx]

    @staticmethod
    def query_path_node(path_df, node_id):
        out = []
        for i in range(1, 4):
            out.append(MFGPathPlotter.query_path_pos(path_df, node_id, i))
        return pd.concat(out).sort_values('model_metric', ascending=False)

    def rank_connecting_nodes(self, path_df, metric='model_metric'):

        first_path = path_df.iloc[0, 0]
        comp_id = first_path[0]
        dis_id = first_path[-1]

        metric_vals = path_df[[metric]].reset_index()

        # Expand all the nodes
        all_nodes = dfp.expand_split_col(path_df['node_ids'])

        # Get the index of each node in the list (this is essential degrees of separation)
        deg_sep = path_df['nodes'].apply(lambda x: list(range(len(x)))).rename('deg_sep')
        deg_sep = dfp.expand_split_col(deg_sep).drop('old_idx', axis=1)
        all_nodes = pd.concat([all_nodes, deg_sep], axis=1)

        # Add in each metric for each instance of a node
        all_nodes = all_nodes.merge(metric_vals, left_on=['old_idx'], right_on=['index']).drop(['index', 'old_idx'], axis=1)

        # Group on the nodes and add the metrics
        node_metrics = all_nodes.groupby('node_ids')[metric].sum()
        # Get the minimum degress of separation
        deg_sep = all_nodes.groupby('node_ids')['deg_sep'].min()

        # Rejoin the metrics and degress of separation and sort.
        all_nodes = pd.concat([node_metrics, deg_sep], axis=1).reset_index().sort_values(metric, ascending=False)

        # Filter out the original compound and disease
        all_nodes = all_nodes.query('node_ids != @comp_id and node_ids != @dis_id').reset_index(drop=True)
        all_nodes['node_name'] = all_nodes['node_ids'].map(self.id_to_name)
        all_nodes['node_label'] = all_nodes['node_ids'].map(self.id_to_label)

        return all_nodes.rename(columns={'node_ids': 'node_id'})


    def _min_max(self, nums):
        this_min = min(nums)
        this_max = max(nums)
        return [(num - this_min) / (this_max - this_min) for num in nums]

    def draw_top_paths(self, path_df, head_num=10, metric='model_metric', poi_idx=None, proba=None):

        top_n_paths = path_df.sort_values(metric, ascending=False).head(head_num)

        # Get the list of paths
        list_of_paths = top_n_paths['node_ids'].tolist()
        path_weights = top_n_paths[metric].tolist()
        # Z_scores are huge and need normalizaion
        if metric == 'z_score':
            path_weights = self._min_max(path_weights)

        # Get the edge names
        list_of_edges = []
        for mp in top_n_paths['metapath']:
            list_of_edges.append(gp.parse_metapath_to_edge_names(mp, self.mp_info, self.inv_map))

        if type(poi_idx) == list:
            ec = []
            for idx in poi_idx:
                poi = path_df.loc[idx, 'node_ids']
                ec.append(gp.highlight_path_of_interest(list_of_paths, poi))

            # Take the edges from the first highlighted poi
            edge_color_map = ec[0]
            for ecm in ec[1:]:
                for k, v in ecm.items():
                    # Any other highlihted paths get marked
                    # seaborn 1 is highlight color.
                    if v == sns.color_palette().as_hex()[1]:
                        edge_color_map[k] = v

        elif poi_idx is not None:
            path_of_interest = path_df.loc[poi_idx, 'node_ids']
            edge_color_map = gp.highlight_path_of_interest(list_of_paths, path_of_interest)
        else:
            path_of_interest = []
            edge_color_map = gp.highlight_path_of_interest(list_of_paths, path_of_interest)



        G = gp.build_explanitory_graph(list_of_paths, list_of_edges, path_weights=path_weights,
                                    node_id_to_label=self.id_to_label, node_id_to_color=self.node_id_to_color,
                                    edge_id_to_color=edge_color_map, min_dist=3)

        gp.draw_explanitory_graph(G, node_id_to_name=self.id_to_name, proba=proba, n_paths=10, xscale=17, title=False);


