class BoxClassifier:
    lookup_matrix = None
    
    def __init__(self, partitions = 20):
        self.partitions = partitions
    
    def fit(self, xtrain, ytrain):
        feature_linspace = [] # will be size of number of features (xtrain.shape[1])

        for i in range(xtrain.shape[1]): # for each feature, compute linspace
            bounds = min(xtrain[:,i]), max(xtrain[:,i])
            feature_linspace.append(np.linspace(bounds[0], bounds[1], self.partitions))

        feature_linspace_averages = []
        avg_vec = []

        for i in range(len(feature_linspace)):
            for x in range(self.partitions - 1):
                avg_vec.append((feature_linspace[i][x] + feature_linspace[i][x+1]) / 2)
            feature_linspace_averages.append(avg_vec)
            avg_vec = []

        xy_train = np.concatenate((xtrain,ytrain.reshape(-1,1)), axis = 1)

        # iterate over data features and compute nearest neighbor
        target_array = []

        mean_array = []
        mean_sub_array = []

        for i in range(xtrain.shape[0]):
            for f in range(xtrain.shape[1]):
                min_index = np.argmin(np.array(abs(feature_linspace_averages[f] - xtrain[i,f])))
                mean_sub_array.append(feature_linspace_averages[f][min_index])
            mean_array.append(mean_sub_array)
            target_array.append(xy_train[i,-1])
            mean_sub_array = []

        self.lookup_matrix = np.array(pd.concat([pd.DataFrame(mean_array), pd.DataFrame(target_array)], axis=1).drop_duplicates())
    
    def predict(self, xtest):
        predictions = []
        for i in range(xtest.shape[0]):
            lookup_values = []
            total_array = np.zeros(self.lookup_matrix.shape[0])
            for f in range(xtest.shape[1]):
                lookup_values.append(abs(self.lookup_matrix[:,f] - xtest[i,f]))
            for f in range(xtest.shape[1]):
                total_array += lookup_values[f]

            predictions.append(int(self.lookup_matrix[np.argmin(total_array),-1]))
        return predictions
    
    # for use with sklearn's cross_val_score
    def get_params(self, deep=False):
        temp_dict = dict()
        temp_dict['partitions'] = self.partitions
        return temp_dict
