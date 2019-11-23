import numpy as np
import math
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement this method
        # ceil[(high-low)/tile_width] + 1
        self.feature_len = 1
        self.dimension_num = []
        for i in range(len(state_low)):
            self.dimension_num.append((math.ceil((state_high[i] - state_low[i]) / tile_width[i]) + 1))
            self.feature_len *= (math.ceil((state_high[i] - state_low[i]) / tile_width[i]) + 1)
        # Dimension of each tiling's feature vector
        self.single_feature_len = self.feature_len
        # Dimension of total tilings' feature vector
        self.feature_len *= num_tilings
        # Weight vector
        self.w = np.zeros(self.feature_len)
        # Number of tilings
        self.num_tilings = num_tilings
        # Other configs
        self.state_low = state_low
        self.state_high = state_high
        self.tile_width = tile_width
        # print("Single feature len: {}".format(self.single_feature_len))
        # print("Total feature len: {}".format(self.feature_len))
        # print("Dimension info: {}".format(self.dimension_num))

    def helper(self, s):
        s_vec = np.zeros(self.feature_len)
        for k in range(self.num_tilings):
            # k-th tilings
            # (low - tiling_index / # tilings * tile width)
            pos = 0
            total_dimension = self.single_feature_len
            idx_list = []
            for i in range(s.shape[0]):
                # i-th dimension
                idx = math.floor((s[i] - (self.state_low[i] - k / self.num_tilings * self.tile_width[i])) / self.tile_width[i])
                total_dimension = int(total_dimension / self.dimension_num[i])
                pos += total_dimension * idx
                idx_list.append(idx)
            # Flatten the axis coordinates into position in 1-dimension vector
            s_vec[pos + self.single_feature_len * k] = 1.0
        return s_vec, np.dot(s_vec, self.w)

    def __call__(self,s):
        # TODO: implement this method
        s_vec, res = self.helper(s)
        return res

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        s_vec, res = self.helper(s_tau)
        self.w += alpha * (G - res) * s_vec
        return None
