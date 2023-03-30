#!/usr/bin/env python3

"""
Eigenanalysis toolbox
object-oriented implementation.

Data examples according to Palaeomath101
https://www.palass.org/publications/newsletter/palaeomath-101

PCA testing done according to 
http://stats.stackexchange.com/questions/102882/steps-done-in-factor-analysis-compared-to-steps-done-in-pca/102999#102999
"""

__author__      = "Falk Mielke"
__date__        = 20180928


#________________________________________________________________________________
### Prerequisites
import numpy as NP
import pandas as PD
import scipy.linalg as LA
import shelve as SH # storage


#######################################################################
### Eigenanalysis                                                   ###
#######################################################################
#________________________________________________________________________________
class Eigenanalysis (object):
    # a parent object/basic API for eigenanalyses (e.g. PCA, FA, ICA...)
    # which will be subclassed below.

    def __init__(self, data: PD.DataFrame \
                 , features: list \
                 , reduce_dim_to: int = None \
                 , component_label: str = 'EA' \
                 , from_dict: bool = False \
                 , standardize: bool = False \
                 ):

        # store input
        self._data = data.copy()
        self._features = features
        self._comp_label = component_label

        # dimensionality
        self.auto_dim = reduce_dim_to == 'auto'
        self.dim = len(features) if ((reduce_dim_to is None) or self.auto_dim) else reduce_dim_to

        # escape if loaded
        if from_dict:
            return

        # update calculated quantities
        self.UpdateMeasures()

        # store initial measures
        self._rawcentroid = self.GetCentroid().copy()
        self._rawstd = self.std.copy()
        self._rawsize = self.rcs.copy()

        # optional: standardization
        self.centered = False
        self.standardized = False
        if standardize:
            self.Standardize()

        # Analysis!
        self.AnalysisProcess()





    def AnalysisProcess(self):
        # this is a basic default implementation
        # and should be overwritten in subclasses
        self.EigenAnalyze(self.correlation)


#________________________________________________________________________________
### data update
    def UpdateMeasures(self):
        self.shape = self[:].shape
        self.centroid = self.GetCentroid()
        self.std = NP.std(self[:], axis = 0, ddof = 1)

        #root centroid size
        self.rcs = NP.sqrt(NP.sum([NP.power(self[:,nr]-self.centroid[nr], 2) for nr in range(len(self._features))]))

        # correlation matrix
        self.correlation = NP.corrcoef(self[:], rowvar = False, ddof = 1)

        # covariance matrix
        self.covariance = NP.cov(self[:], rowvar = False, ddof = 1)

#________________________________________________________________________________
### Sandardization
    def Center(self):
        # print ("/* ignore warning")
        self[:] -= self.centroid
        # print ("ignore warning */")
        self.UpdateMeasures()
        self.centered = True

    def Scale(self):
        self[:] /= self.rcs
        self.UpdateMeasures()

    def Standardize(self):
        # center and scale to Normal
        # _raw... quantities must not be touched.

        if self.standardized:
            # ... is already standardized
            # (though repeat should not matter)
            return
        self[:] = (self[:] - self.centroid) / self.std
        self.UpdateMeasures()
        self.centered = True
        self.standardized = True
       

# perform Eigenanalysis
    def EigenAnalyze(self, matrix, real = True):
        eigenvalues, eigenvectors = LA.eig(matrix)

        idx = sorted(range(len(eigenvalues)), key=lambda k: eigenvalues[k], reverse=True)
        self.vectors = [eigenvectors[:, nr] for nr in range(eigenvectors.shape[1])]
        self.vectors = [NP.real(self.vectors[i]) if real else self.vectors[i] for i in idx]
        self.values = NP.real(eigenvalues[idx]) if real else eigenvalues[idx]
        self.weights = self.values / NP.sum(self.values)
        self.dim = len(self.values)

        self.variance_conserved = NP.sum(self.weights)

        # reduce dimension
        if self.auto_dim:
            self.ReduceDimensionality('auto', threshold = 0.95)

        self.UpdateTransformed()



    def UpdateTransformed(self):
        # transform data
        self.transformed = PD.DataFrame(  self.Transform(self[:].copy()) \
                                        , columns = self.GetColumns() \
                                        , index = self._data.index \
                                        )


    def ReduceDimensionality(self, redu2dim, threshold = 0.95):
        self.dim = (NP.argmax(NP.cumsum(self.weights) > threshold) + 1) if redu2dim == 'auto' else redu2dim

        self.variance_conserved = NP.sum(self.values[:self.dim]) / NP.sum(self.values)
        self.values = self.values[:self.dim]
        self.vectors = self.vectors[:self.dim]
        self.weights = self.values / NP.sum(self.values)

        self.UpdateTransformed()




#________________________________________________________________________________
### data transformations
    def Transform(self, data):
        return data.dot(NP.array(self.vectors[:]).T)


    def TraFo(self, data):
        data = data.copy()
        if self.centered or self.standardized:
            data -= self._rawcentroid
        if self.standardized:
            data /= self._rawstd
        return PD.DataFrame(self.Transform(data.values) \
                            , columns = self.GetColumns() \
                            , index = data.index \
                            )

    def ReTraFo(self, component_scores, unshift = True, unscale = True):
        retrafo = NP.dot(component_scores, NP.array(self.vectors))
        if self.standardized:
            retrafo *= self._rawstd.T if unscale else 1.
        if self.centered or self.standardized:
            retrafo += self._rawcentroid if unshift else self.centroid

        return retrafo

    def InverseTransform(self, pc_scores, raw_mean = None, raw_scale = None):
        # retrafo with different scaling
        retransformed_values = NP.dot(pc_scores, NP.array(self.vectors))
        if raw_scale is not None:
            retransformed_values *= raw_scale
        if raw_mean is not None:
            retransformed_values += raw_mean

        return retransformed_values


#________________________________________________________________________________
### some extra algebra
    def GetColumns(self):
        return [f'{self._comp_label}{nr+1}' for nr in range(len(self.values))]

    def GetLoadings(self, as_df = False):
        loadings = [self.vectors[i]*NP.sqrt(self.values[i]) for i in range(len(self.values))]

        if as_df:
            return PD.DataFrame(NP.stack(loadings) \
                                , columns = self._features \
                                , index = self.GetColumns() \
                                ).T

        return loadings

    def GetCommunalities(self):
        loadings = NP.stack(self.GetLoadings())
        return NP.sum(NP.power(loadings, 2), axis = 0)

    def GetRegressionCoefficients(self):
        return NP.dot(NP.stack(self.GetLoadings()).T, NP.eye(len(self.values)) * 1/self.values) 

    def GetStandardizedScores(self, values):
        CentralNorm = lambda vec: (vec - NP.mean(vec, axis = 0)) / NP.std(vec, axis = 0)
        return CentralNorm(NP.dot(values, self.GetRegressionCoefficients()))

    def GetCentroid(self):
        return NP.mean(self[:], axis = 0)


#________________________________________________________________________________
### Storage
    def ToDict(self):
        return dict( \
              vectors = self.vectors \
            , values = self.values \
            , weights = self.weights \
            , dim = self.dim \
            , _rawcentroid = self._rawcentroid \
            , _rawstd = self._rawstd \
            , _rawsize = self._rawsize \
            , _features = self._features \
            , _data = self._data \
            , variance_conserved = self.variance_conserved \
            , transformed = self.transformed  \
            , centered = self.centered \
            , standardized = self.standardized \
            )

    @classmethod
    def Load(cls, load_filename):
        # load segment
        loaded = {}
        with SH.open(load_filename) as store:
            for key in store.keys():
                loaded[key] = store[key]

        return cls.FromDict(loaded)

    @classmethod
    def FromDict(cls, loaded):
        new_pca = cls(data = loaded['_data'] \
                      , features = loaded['_features'] \
                      , reduce_dim_to = loaded['dim'] \
                      , from_dict = True \
                      )

        new_pca.vectors = loaded['vectors']
        new_pca.values = loaded['values']
        new_pca.weights = loaded['weights']
        # new_pca.dim = loaded['dim']
        new_pca._rawcentroid = loaded['_rawcentroid']
        new_pca._rawstd = loaded['_rawstd']
        new_pca._rawsize = loaded['_rawsize']
        # new_pca._features = loaded['_features']
        # new_pca._data = loaded['_data']
        new_pca.variance_conserved = loaded['variance_conserved']
        new_pca.transformed = loaded['transformed']
        new_pca.centered = loaded['centered']
        new_pca.standardized = loaded['standardized']
        # new_pca.centered = loaded.get('centered', NP.sum(NP.abs(new_pca.GetCentroid())) < 1e-3)
        # new_pca.standardized = loaded.get('standardized', NP.sum(NP.abs(new_pca.GetCentroid())) < 1e-3)

        new_pca.UpdateMeasures()
        return new_pca

    def Copy(self):
        self_dict = self.ToDict()
        return Eigenanalysis.FromDict(self_dict.copy())

    def Save(self, store_filename):
        # save a PCA to shelf
        print (f"saving {store_filename}", " "*16, end = '\r', flush = True)

        with SH.open(store_filename) as store:
            # store all keys
            for key, value in self.ToDict().items():
                store[key] = value

        print (f"saving {store_filename} done!", " "*16, end = '\r', flush = True)


#________________________________________________________________________________
### administriative functions
    # http://rafekettler.com/magicmethods.html

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, point):
        # if key is of invalid type or value, the list values will raise the error
        return NP.array(self._data.loc[:, self._features])[point]

    def __setitem__(self, point, value):
        cache = NP.array(self._data.loc[:, self._features])
        cache[point] = value
        self._data.loc[:, self._features] = cache

    def __delitem__(self, item, column = None):
        if column is None:
            self._data.drop(item, axis=0)
        else:
            self._data.drop(self._data[self._data[column] == item].index, axis = 0)

    def __iter__(self):
        return iter(NP.array(self._data.loc[:, self._features]))


    def __str__(self):
        text = ["ev%i (%2.1f%%): %.2f; [%s]" % (   nr \
                                                , self.weights[nr]*100 \
                                                , self.values[nr] \
                                                , ",".join(list(map( lambda vec: "%.2f" % (vec), self.vectors[nr]))) \
                                                ) \
                for nr in range(len(self.values))]

        eigen_str = "\t\t%s\n" % (','.join(self._features))+"\n".join(text) \
                + ("" if self.variance_conserved == 1 else "\n(%.2f%% variance conserved)" % (self.variance_conserved*100))

        return eigen_str + "\n" + str(self.transformed.head(5))+"\n\t..."

    def ToORG(self):
        # format as an org table
        table = PD.DataFrame( self.vectors \
                              , columns = self._features \
                              , index = [f'PC{nr+1} (\({self.weights[nr]*100:.1f} \\%\))' \
                                        for nr in range(len(self.values)) \
                                        ] \
                            )
        return table


    ## arithmetics
    # summation
    def __add__(self, value):
        return self._data.loc[:, self._features] + value

    def __iadd__(self, value):
        self._data.loc[:, self._features] += value
        self.UpdateMeasures()
        return self

    # subtraction
    def __sub__(self, value):
        return self._data.loc[:, self._features] - value

    def __isub__(self, value):
        self._data.loc[:, self._features] -= value
        self.UpdateMeasures()
        return self

    # multiplication
    def __mul__(self, value):
        return self._data.loc[:, self._features] * value

    def __imul__(self, value):
        self._data.loc[:, self._features] *= value
        self.UpdateMeasures()
        return self

    # division
    def __div__(self, value):
        return self._data.loc[:, self._features] / value

    def __idiv__(self, value):
        self._data.loc[:, self._features] /= value
        self.UpdateMeasures()
        return self

    # division
    def __pow__(self, value):
        return self._data.loc[:, self._features] ** value

    def __ipow__(self, value):
        self._data.loc[:, self._features] **= value
        self.UpdateMeasures()
        return self





#######################################################################
### Principal Component Analysis                                    ###
#######################################################################
#________________________________________________________________________________
class PrincipalComponentAnalysis(Eigenanalysis):
### load data and perform PCA
    def __init__(self, data, features, reduce_dim_to = None, **kwargs):
        super(PrincipalComponentAnalysis, self).__init__(data, features, reduce_dim_to, component_label = 'PC', **kwargs)


### PCA process
    def AnalysisProcess(self):
    # center
        self.Center()
    # perform eigenanalysis of covariancematrix
        self.EigenAnalyze(self.covariance)

### cp
    def Copy(self):
        self_dict = self.ToDict()
        return PrincipalComponentAnalysis.FromDict(self_dict.copy())



#######################################################################
### Phylogenetic Principal Component Analysis                       ###
#######################################################################
# https://www.researchgate.net/figure/Transformation-from-a-phylogenetic-tree-to-a-variance-covariance-matrix-under-the_fig1_228086346
def GetPhylogeneticRelatedness(tree):
    # Transformation from a phylogenetic tree to a variance-covariance matrix under the Brownian Motion (BM) model: 
    #   the variance is set to be the branch length from the root to the tip. 
    #   The covariance is the branch length from the root to the most recent common ancestor.

    rt = tree.get_tree_root()

    leaves = tree.get_leaf_names()
    n_leaves = len(leaves)
    leaf_ids = {lv: nr for nr, lv in enumerate(leaves)}

    shared_branch = PD.DataFrame(NP.zeros([n_leaves, n_leaves]), columns = tree.get_leaf_names(), index = tree.get_leaf_names())
    for leaf1 in leaves:
        if len(leaves) > 100:
            print ("%.1f %%" % (100 * leaf_ids[leaf1] / len(leaves) ), end='\r', flush=True)
        for leaf2 in leaves:
            if leaf_ids[leaf2] < leaf_ids[leaf1]:
                # only traverse half the leaves
                continue

            n1 = tree.get_leaves_by_name(leaf1)[0]
            ca = n1.get_common_ancestor(tree.get_leaves_by_name(leaf2)[0])
            # note that, if the leaf is itself, then trivially the common ancestor is itself.
            
            dist = tree.get_distance(rt, ca, topology_only = False)
                
            shared_branch.loc[leaf1, leaf2] = shared_branch.loc[leaf2, leaf1] = dist

    # http://firsttimeprogrammer.blogspot.com/2015/01/how-to-build-variance-covariance-matrix.html
    #, shared_branch.cov()
    return shared_branch 

#________________________________________________________________________________
class PhylogeneticPrincipalComponentAnalysis(Eigenanalysis):
### load data and perform PCA
    def __init__(self, data, features, tree, reduce_dim_to = None):


        tree_groups = list(sorted([leaf for leaf in tree.get_leaf_names() if leaf in data.index.values]))
        tree.prune(tree_groups)
        if not (data.shape[0] == len(tree_groups)):
            raise IOError("data indices (%i) do not match tree groups (%i). (pruned? averaged? unique?)" % (data.shape[0], len(tree_groups)))

        # expected covariances of our data due to phylogenetic relatedness 
        #   i.e. matrix of shared branch lengths
        shared_branchlengths = GetPhylogeneticRelatedness(tree)

        data_matrix = data.loc[tree_groups, features].values
       
        phylogenetic_covariance = shared_branchlengths.loc[tree_groups, tree_groups].values
        phylogenetic_covariance_inv = LA.inv(phylogenetic_covariance)

        phylogenetic_means = NP.sum(phylogenetic_covariance_inv.dot(data_matrix), axis = 0)/NP.sum(phylogenetic_covariance_inv, axis = (0,1))

        phylogenetic_means_matrix = NP.stack([phylogenetic_means]*len(tree_groups), axis = 0)

        data_difference_from_phymean = data_matrix - phylogenetic_means_matrix


        self.evol_varcovar = (1/(len(tree_groups)-1))*(data_difference_from_phymean.T.dot(phylogenetic_covariance_inv).dot(data_difference_from_phymean))

        super(PhylogeneticPrincipalComponentAnalysis, self).__init__(data, features, reduce_dim_to, component_label = 'pPC')


### PCA process
    def AnalysisProcess(self):
    # center
        self.Center()
    # perform eigenanalysis of covariancematrix
        self.EigenAnalyze(self.evol_varcovar)


#######################################################################
### Between Group Principal Component Analysis                      ###
#######################################################################
#________________________________________________________________________________
class BetweenGroupPrincipalComponentAnalysis(Eigenanalysis):
### load data and perform PCA
    def __init__(self, data, features, groups, reduce_dim_to = None):
        self._group_columns = groups
        for grpcol in self._group_columns:
            data[grpcol] = PD.Categorical(data[grpcol].values, ordered = False)
        
        self._observations = data.copy()
        data = data.loc[:, self._group_columns+features].groupby(self._group_columns).agg(NP.mean)#.reset_index(inplace = False)
        self.groups = data.index.values

        super(BetweenGroupPrincipalComponentAnalysis, self).__init__(data, features, reduce_dim_to, component_label = 'bgPC')


### PCA process
    def AnalysisProcess(self):
    # center
        self.Center()
    # perform eigenanalysis of covariancematrix
        self.EigenAnalyze(self.covariance)


    def UpdateTransformed(self):
        # transform data
        self.transformed = PD.DataFrame(  self.Transform(self[:].copy()) \
                                        , columns = self.GetColumns() \
                                        , index = self._data.index \
                                        )

        obs_trafo = self.Transform(self._observations.loc[:,self._features].values - self._rawcentroid) 
        data_loadings = PD.DataFrame(obs_trafo)
        data_loadings.columns = self.GetColumns()
        for col in self._group_columns:
            data_loadings[col] = self._observations.loc[:, col].values
        data_loadings.index = self._observations.index

        self._observations_trafo = data_loadings
        self._observations_trafo.index = self._observations.index


    def Plot(self, component_selection = [[1,2],[3,2],[1,3],[4,5]], subplots = [2,2], colors = None, families = None):
        import matplotlib as MP
        import matplotlib.pyplot as MPP

        if families is not None:
            from svgpathtools import svg2paths
            import os as OS

            silhouettes = {}
            icon_folder = 'figures/icons'
            icons = [OS.path.splitext(fi)[0] for fi in OS.listdir(icon_folder)]
            print (icons)

            for family in icons:
                paths, attributes = svg2paths('%s/%s.svg' % (icon_folder,family))
                animal = []
                for path in paths:
                    for elm in path:
                        animal.append([-NP.real(elm.start), -NP.imag(elm.start)])
                        animal.append([-NP.real(elm.end), -NP.imag(elm.end)])

                silhouettes[family] = NP.array(animal)
                silhouettes[family] -= NP.mean(silhouettes[family], axis = 0)

            print ([k for k in silhouettes.keys()])

        fig = MPP.figure(figsize = (48/2.54, 36/2.54))
        fig.subplots_adjust(top = .98, right  = .98, bottom = .06, left = .06, wspace = .10, hspace = .10)

        print (self.groups)
        if colors is None:
            lm = NP.arange(len(self.groups))
            mpl_colormap = MP.cm.ScalarMappable(norm=MP.colors.Normalize(vmin=lm.min(), vmax=lm.max()), cmap=MPP.get_cmap('Dark2') ) # Accent, Dark2, Set3
            
            colors = { group_key: mpl_colormap.to_rgba(nr) for nr, group_key in enumerate(self.groups) }

        # print (colors)

        for ax_nr, pc_axes in enumerate(component_selection):
            ax = fig.add_subplot(subplots[0],subplots[1],ax_nr+1)
            components = ['bgPC%i'%(pc) for pc in pc_axes]

            for grp, values in self.transformed.iterrows():

                row_indexer = []
                for grpcol_nr, grpcol_value in enumerate([grp]):
                    grpcol = self._group_columns[grpcol_nr]
                    row_indexer.append(self._observations_trafo[grpcol].values == grpcol_value)
                row_indexer = NP.all(NP.stack(row_indexer, axis = 1), axis = 1)

                point_values = self._observations_trafo.loc[row_indexer, components].values 
                mean_values = values[components]

                if families is not None:
                    tester = self._observations_trafo.loc[row_indexer, 'group'].index.values[0]
                    # print (tester, families[(tester[0], tester[1])], grp)

                    ax.scatter(mean_values[0], mean_values[1], s = 1200, marker = silhouettes[families[(tester[0], tester[1])]], edgecolor = '0', alpha = 0.75, facecolor = colors[grp], zorder = 20, label = None)#grp)
                else:
                    # ax.scatter(point_values[:,0], point_values[:,1], s = 40, marker = 'o', edgecolor = 'none', facecolor = colors[grp], alpha = 0.3, zorder = 10)
                    ax.scatter(mean_values[0], mean_values[1], s = 80, marker = 'o', edgecolor = '0', facecolor = colors[grp], zorder = 20, label = None)#grp)
                # if grp in [   'Giraffa camelopardis' \
                #             , 'Alces alces' \
                #             , 'Bos taurus' \
                #             , 'Lama glama' \
                #             , 'Equus caballus' \
                #             , 'Tapirus indicus' \
                #             ]:
                # if grp in [   'Giraffa' \
                #             # , 'Rhinoceros' \
                #             # , 'Diceros' \
                #             # , 'Dicerorhinus' \
                #             # , 'Ceratotherium' \
                #             # , 'Lama' \
                #             # , 'Camelus' \
                #             # , 'Taurotragus' \
                #             # , 'Tapirus' \
                #             # , 'Alces' \
                #             # , 'Bos' \
                #             # , 'Equus' \
                #             # , 'Tragelaphus' \
                #             ]:
                #     print (grp)
                #     print (self._observations_trafo.loc[row_indexer, components])
                #     ax.scatter(point_values[:,0], point_values[:,1], s = 20, marker = 'o', edgecolor = 'k', facecolor = colors[grp], alpha = 0.8, zorder = 30)

                ax.annotate( grp \
                            , xy = (mean_values[0], mean_values[1]) \
                            , xycoords = 'data' \
                            , xytext = (mean_values[0], mean_values[1]) \
                            , textcoords = 'data' \
                            , fontsize = 6 \
                            , verticalalignment = 'bottom', horizontalalignment = 'left' \
                            , zorder = 24 \
                            )
                

                ax.set_xlabel('bgPC%i (%.1f %%)'%(pc_axes[0], 100 * self.weights[pc_axes[0]-1]))
                ax.set_ylabel('bgPC%i (%.1f %%)'%(pc_axes[1], 100 * self.weights[pc_axes[1]-1]))
            # if ax_nr == len(component_selection)-1:
            #     ax.legend(loc = 'best', fontsize = 6, ncol = 2)

        # MPP.show()


#######################################################################
### Principal Component Analysis                                    ###
#######################################################################
#________________________________________________________________________________
class SubsurfacePCA(Eigenanalysis):
######## UNUSED! ####################
# (scaling does not make sense)
### load data and perform PCA
    def __init__(self, data, features, subsurfaces, reduce_dim_to = None, scaling = False):
        self.subsurfaces = subsurfaces
        self.scaling = scaling
        super(SubsurfacePCA, self).__init__(data, features, reduce_dim_to, component_label = 'PC')

### PCA process
    def AnalysisProcess(self):
    # center
        print (self._data)
        self.Center()
        print (self._data)
        if self.scaling:
            self.Scale()

        print (self._data)

        self.UpdateMeasures()

    # perform eigenanalysis of covariancematrix
        self.EigenAnalyze(self.covariance)

#________________________________________________________________________________
### data update
    def UpdateMeasures(self):
        self.shape = self[:].shape
        self.centroid = NP.mean(self[:], axis = 0)
        self.std = NP.std(self[:], axis = 0, ddof = 1)

        #root centroid size
        self.rcs = NP.sqrt(NP.sum([NP.power(self[:,nr]-self.centroid[nr], 2) for nr in range(len(self._features))]))

        # correlation matrix
        self.correlation = NP.corrcoef(self[:], rowvar = False, ddof = 1)

        # covariance matrix
        self.covariance = NP.cov(self[:], rowvar = False, ddof = 1)


    def Center(self):
        ### scale
        self._rawcentoid = NP.mean(self._data.loc[:, self._features].values, axis = 0).ravel()
        self._data.loc[:, self._features] -= NP.ones((self._data.shape[0], len(self._features))) * self._rawcentroid
        

    def Scale(self):
        ### scale
        self._rawscale = NP.std(self._data.loc[:, self._features].values, axis = 0, ddof = 1).ravel()
        # self._rawscale[0] = 0.1

        # TODO: somehow scaling fails.
        print (NP.ones((self._data.shape[0], len(self._features))) * self._rawscale)
        print (self._data.loc[:, self._features].shape, self._rawscale.shape)
        self._data.loc[:, self._features] = NP.divide( self._data.loc[:, self._features] \
                                                     , NP.ones((self._data.shape[0], len(self._features))) * self._rawscale \
                                                     )
        

    def TransformObservations(self, observations_raw):
        observations = observations_raw.copy()
        # ## shift
        observations.loc[:,self._features] -= NP.ones((observations.shape[0], len(self._features))) * self._rawcentroid
        # observations.loc[:,self._features] -= NP.ones((observations.shape[0], len(self._features))) * self.centroid

        ## scale
        if self.scaling:
            observations.loc[:, self._features] /= NP.ones((observations.shape[0], len(self._features))) * self._rawscale
            # observations.loc[:, self._features] /= NP.ones((observations.shape[0], len(self._features))) * self.GetStDev()


        ## transform
        trafo_observations = PD.DataFrame(self.TraFo(observations.loc[:, self._features].values), index = observations.index)
        trafo_observations.columns = self.GetColumns()
        trafo_observations = trafo_observations.join(observations.loc[:,['group_nr']])


        return trafo_observations







#######################################################################
### Factor Analysis                                                 ###
#######################################################################
#________________________________________________________________________________
class FactorAnalysis(Eigenanalysis):
### load data and perform PCA
    def __init__(self, data, features, reduce_dim_to = None):
        super(FactorAnalysis, self).__init__(data, features, reduce_dim_to, component_label = 'F')


### PCA process
    def AnalysisProcess(self):
    # center
        # self.Center()
        print (self.correlation)
    # perform eigenanalysis of covariancematrix
        self.EigenAnalyze(self.correlation)


## TODO: transform by standardized scores!


#######################################################################
### Canonical Variate Analysis                                      ###
#######################################################################
#________________________________________________________________________________
class CanonicalVariateAnalysis(Eigenanalysis):
### load data and perform CVA
    def __init__(self, data, features, groups, reduce_dim_to = None):
        self._groups = groups # needs to be a list of columns in the data
        super(CanonicalVariateAnalysis, self).__init__(data, features, reduce_dim_to, component_label = 'CV')

### CVA process        
    def AnalysisProcess(self):
        self.Center()

        deviation_localmean = self._data.groupby(self._groups).transform(lambda x: x - x.mean())
        local_covar = NP.cov(deviation_localmean, rowvar = False, ddof = 1)

        between_covar = NP.cov(self[:], rowvar = False, ddof = 1) - local_covar

        self.EigenAnalyze(NP.dot(LA.inv(local_covar), between_covar))


    def UpdateTransformed(self):
        # transform data
        self.transformed = PD.DataFrame(  self.Transform(self[:]) \
                                        , columns = self.GetColumns() \
                                        , index = self._data.index \
                                        )

        self.transformed[self._groups] = self._data[self._groups]


#######################################################################
### Independent Component Analysis                                    ###
#######################################################################
#________________________________________________________________________________
class IndependentComponentAnalysis(Eigenanalysis):
### load data and perform PCA
    def __init__(self, data, features, reduce_dim_to = None):
        if reduce_dim_to == 'auto':
            print ("auto dim reduction not useful for ICA!")
            return None
        self.auto_dim = False
        self.n_components = reduce_dim_to

        super(IndependentComponentAnalysis, self).__init__(data, features, reduce_dim_to = None, component_label = 'IC')

    def ReduceDimensionality(self, redu2dim, threshold = 0.95):
        pass

### PCA process
    def AnalysisProcess(self):
    # center
        self._rawcentroid = self.centroid
        self._rawsize = self.std
        self.Standardize()
    # perform eigenanalysis of covariancematrix
        self.EigenAnalyze(self._data.loc[:,self._features].values)


# perform Eigenanalysis
    def EigenAnalyze(self, matrix, real = True):

        import sklearn.decomposition as DEC
        ica = DEC.FastICA(n_components = self.n_components, max_iter = 10000, tol = 1e-6)
        ica.fit(matrix)
        self._ica = ica
        eigenvectors = ica.mixing_.copy()
        eigenvalues = NP.sum(NP.abs(eigenvectors), axis = 0)

        # NormedEV = lambda vec: NP.divide(vec, NP.sqrt(NP.sum(NP.power(vec,2))))
        NormedEV = lambda vec: vec
        idx = sorted(range(len(eigenvalues)), key=lambda k: eigenvalues[k], reverse=True)
        self.vectors = [NormedEV(eigenvectors[:, nr]) for nr in range(eigenvectors.shape[1])]
        # self.vectors = [NP.real(self.vectors[i]) if real else self.vectors[i] for i in idx]
        self.values = NP.real(eigenvalues[idx]) if real else eigenvalues[idx]
        self.weights = self.values / NP.sum(self.values)
        self.dim = len(self.values)

        self._ica.mixing_ = NP.stack(self.vectors, axis = 1)

        self.variance_conserved = NP.sum(self.weights)

        self.UpdateTransformed()

    def Transform(self, data):
        return self._ica.transform(data)
        # return data.dot(LA.pinv(NP.array(self.vectors)).T)


    def ReTraFo(self, component_scores, unshift = True):
        if len(component_scores.shape) == 1:
            component_scores = component_scores.reshape(1,-1)
        # retrafo = component_scores.values.dot(NP.array(self.vectors).T)
        retrafo = self._ica.inverse_transform(component_scores)
        retrafo *= self._rawsize.T if unshift else 1
        retrafo += self._rawcentroid if unshift else self.centroid
        return retrafo

#######################################################################
### Testing Processes                                               ###
#######################################################################
#________________________________________________________________________________
def TestManual(values, feature_names, dim):
    if dim == 'auto':
        dim = 3
### step-by-step manual procedure
    # !! data must be centered for PCA
    # center 
    raw_means = NP.mean(values, axis = 0)
    values -= raw_means
    # standardize
    if False:
        values /= NP.std(values, axis = 0)


    # covariance matrix
    S = NP.cov(values, rowvar = False, ddof = 1)

    # eigen-decomposition
    eigenvalues, eigenvectors = LA.eig(S)
    eigenvalues = NP.real(eigenvalues)
    print (eigenvalues)

    eigenvectors = {'pc%s' % (nr+1): eigenvectors[:,nr] for nr in range(eigenvectors.shape[1])}
    eigenvectors = PD.DataFrame(eigenvectors)
    eigenvectors['variable'] = feature_names

    print (eigenvectors)


    # check
    import sklearn.decomposition as DEC
    pca = DEC.PCA(n_components=4)
    trafo = pca.fit_transform(values)
    print ("\nScikitlearn Confirmation\n", pca.components_.T)# "\n", pca.explained_variance_ratio_)
    

    # reduce dimensionality
    L = eigenvalues[:dim]
    components = ['pc%s'%(nr+1) for nr in range(dim)]
    V = eigenvectors.loc[:, components].values

    # eigenvector loadings
    A = V * NP.sqrt(L)
    print ("\nLoadings A\n", A)


    # Regression Coefficient B
    B = NP.dot(A, NP.eye(len(L))*1/L)
    print ("\nRegression B\n", B)
    # print (NP.dot(LA.inv(S), A))

    # standardized component scores
    CentralNorm = lambda vec: (vec - NP.mean(vec, axis = 0)) / NP.std(vec, axis = 0)
    scs = NP.dot(values, B)
    scs = PD.DataFrame(CentralNorm(scs), columns = components)
    print ("\nStand. Component Scores\n", scs.head(5),"\n\t...")


    # raw component scores
    rcs = NP.dot(values, V)
    rcs = PD.DataFrame(rcs, columns = components) #  - NP.mean(rcs, axis = 0)
    print ("\nRaw Component Scores\n", rcs.head(5),"\n\t...")

    # # revert transformation
    # dat = NP.dot(rcs.loc[:,:].values, V.T)
    # print (dat + raw_means)


    # scikit learn checks
    trafo = PD.DataFrame(trafo[:,:dim], columns = components)
    print ("\nscikit learn checks\n", trafo.head(5))

    print ("\ntest reversion")
    print ("original:\n", values[:3,:]+raw_means)
    print ("retrafo:\n", NP.dot(NP.array(rcs.loc[:dim,components]), V.T) + raw_means)

#________________________________________________________________________________
def TestPCA(data, feature_names, dim):
# PCA object example usage
    pca = PrincipalComponentAnalysis( \
                                      data \
                                    , features = feature_names \
                                    , reduce_dim_to = dim \
                                    )

    print (pca)
    print (pca.ReTraFo(pca.transformed)[:5,:])
    # print (pca.GetRegressionCoefficients())
    # print (pca.GetStandardizedScores(data.loc[:,feature_names])[:5,:])



#######################################################################
### Testing Procedures                                              ###
#######################################################################
#________________________________________________________________________________
def IrisTestPCA():
    features, data = IrisData()

    # dimensionality reduction
    dim = 4#'auto'#

    # testing
    # only 'setosa' data
    # TestManual(NP.array(data.loc[data['species'] == 'setosa', features]) \
    #             , features, dim)
    # TestPCA(data.loc[data['species'] == 'setosa',:], features, dim)
    TestPCA(data, features, dim)

def IrisTestICA():
    features, data = IrisData()

    # dimensionality reduction
    # PCA object example usage
    ica = IndependentComponentAnalysis( \
                                      data.copy() \
                                    , features = features \
                                    )

    print (ica)
    print (ica.ReTraFo(ica.transformed)[:5,:])
    print(data.iloc[:5,:].loc[:,features].values)

#________________________________________________________________________________
def TrilobiteFA():
    features, data = TrilobiteData()

    fa = FactorAnalysis( \
                          data.loc[:,['species'] + features[:-1]] \
                        , features = features[:-1] \
                        , reduce_dim_to = 'auto' \
                        )

    print (fa)
    print ('loadings:\n\t', fa.GetLoadings())
    print ('communalities:\n\t',fa.GetCommunalities())
    

#________________________________________________________________________________
def IrisTestCVA():
    features, data = IrisData()

    # reduce dataset
    # data = data.groupby('species').head(25).reset_index(drop = True)

    # test!
    cva = CanonicalVariateAnalysis( \
                                      data \
                                    , features = features \
                                    , groups = ['species'] \
                                    )


    cva.ReduceDimensionality('auto', threshold = 0.999)

    print (cva)

    import matplotlib.pyplot as MPP
    import matplotlib.colors as colors
    import matplotlib.cm as cmx


    fig = MPP.figure()
    ax = fig.add_subplot(1,1,1)

    trafo = cva.transformed
    groups = trafo.loc[:,cva._groups].values
    group_map = {grp:nr for nr, grp in enumerate(NP.unique(groups))}
    lm = NP.array(list(group_map.values()))
    colormap = cmx.ScalarMappable(norm=colors.Normalize(vmin=lm.min(), vmax=lm.max()), cmap=MPP.get_cmap('jet') )

    trafo['color_group'] = [group_map[grp] for grp in groups.reshape(len(trafo.index))]

    ax.scatter( \
                  -trafo['CV1'].values \
                , trafo['CV2'].values \
                , s = 80 \
                , marker = 'o' \
                , facecolor = colormap.to_rgba(trafo['color_group'].values) \
                , zorder = 0 \
                , )

    MPP.show()


#________________________________________________________________________________
def IrisTestBGPCA():
    features, data = IrisData()

    # test!
    bgpca = BetweenGroupPrincipalComponentAnalysis( \
                                      data \
                                    , features = features \
                                    , groups = ['species'] \
                                    )

    print (bgpca)

    bgpca.ReduceDimensionality('auto', threshold = 0.999)

    bgpca.Plot(component_selection = [[1,2]], subplots = [1,1])


#######################################################################
### Example Datasets                                                ###
#######################################################################

def TrilobiteData():
    features = ["genus", "body length (mm)", "glabellar length (mm)", "glabellar width (mm)", "eye spacing (mm)"]
    data = PD.DataFrame({features[nr]: col \
                 for nr, col in enumerate( \
                zip(* (\
                  ['Acaste'         , 23.14 , 3.50  , 3.77  , 10.58] \
                , ['Balizoma'       , 14.32 , 3.97  , 4.08  , 7.53 ] \
                , ['Calymene'       , 51.69 , 10.91 , 10.72 , 18.19] \
                , ['Ceraurus'       , 21.15 , 4.90  , 4.69  , 10.63] \
                , ['Cheirurus'      , 31.74 , 9.33  , 12.11 , 13.99] \
                , ['Cybantyx'       , 36.81 , 11.35 , 10.10 , 17.44] \
                , ['Cybeloides'     , 25.13 , 6.39  , 6.81  , 12.49] \
                , ['Dalmanites'     , 32.93 , 8.46  , 6.08  , 13.23] \
                , ['Delphion'       , 21.81 , 6.92  , 9.01  , 10.62] \
                , ['Ormathops'      , 13.88 , 5.03  , 4.34  , 6.46 ] \
                , ['Phacopdina'     , 21.43 , 7.03  , 6.79  , 10.79] \
                , ['Phacops'        , 27.23 , 5.30  , 8.19  , 11.03] \
                , ['Placopoaria'    , 38.15 , 9.40  , 8.71  , 18.11] \
                , ['Pricyclopyge'   , 40.11 , 14.98 , 12.98 , 14.12] \
                , ['Ptychoparia'    , 62.17 , 12.25 , 8.71  , 24.25] \
                , ['Rhenops'        , 55.94 , 19.00 , 13.10 , 23.15] \
                , ['Sphaerexochus'  , 23.31 , 3.84  , 4.60  , 5.27 ] \
                , ['Toxochasmops'   , 46.12 , 8.15  , 11.42 , 16.06] \
                , ['Trimerus'       , 89.43 , 23.18 , 21.52 , 24.18] \
                , ['Zacanthoides'   , 47.89 , 13.56 , 11.78 , 17.81] \
                ) )  \
                )} )

    return features[1:], data



def IrisData():
    
    from sklearn import datasets
    raw = datasets.load_iris()
    data = PD.DataFrame(raw['data'], columns = raw['feature_names'])
    data['species'] = raw['target_names'][raw['target']]

    return raw['feature_names'], data




#######################################################################
### Main Procedure                                                  ###
#######################################################################
if __name__ == "__main__":
    pass
    # IrisTestPCA()
    # IrisTestCVA()
    # IrisTestICA()
    # TrilobiteFA()
    # IrisTestBGPCA()


### /eof. Thanks for Reading!
