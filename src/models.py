import numpy as np
import time

class PopularityRecommender:
    """
    Baseline Model: Recommends items based on their average rating.
    If an item has no ratings in the training set, it returns the Global Mean.
    """
    def __init__(self):
        self.product_mean_ratings = {}
        self.global_mean = 0.0

    def fit(self, data):
        """
        Calculates mean rating for each product.
        Args: train_data (np.ndarray): Matrix of shape (N, 6). Columns: [User, Product, Rating, Time, Year, Weight]
        """
        prod_ids = data[:, 1].astype(int)
        ratings = data[:, 2]

        self.global_mean = np.mean(ratings)

        unique_products, counts = np.unique(prod_ids, return_counts=True)
        max_id = prod_ids.max()
        sum_ratings = np.zeros(max_id + 1)
        np.add.at(sum_ratings, prod_ids, ratings)

        relevant_sums = sum_ratings[unique_products]
        mean_ratings = relevant_sums / counts

        self.product_mean_ratings = dict(zip(unique_products, mean_ratings))
    
    def predict(self, user_id, product_id):
        """Returns item mean if exists, else global mean."""
        return self.product_mean_ratings.get(product_id, self.global_mean)
    
    def evaluate(self, test_data):
        """
        Evaluates the model using RMSE.
        Args: test_data (np.ndarray): Matrix of shape (N, 6). Columns: [User, Product, Rating, Time, Year, Weight]
        Returns: RMSE value
        """
        true_ratings = test_data[:, 2]
        pred_ratings = np.array([self.predict(user, product) for user, product in zip(test_data[:, 0], test_data[:, 1])])
        
        rmse = np.sqrt(np.mean((true_ratings - pred_ratings) ** 2))
        return rmse
    

class MatrixFactorizationRecommender:
    """
    Model: Matrix Factorization with Bias & Time Decay Weighting.
    Algorithm: Stochastic Gradient Descent (SGD).
    """
    def __init__(self, n_users, n_products, n_factors=10, learning_rate=0.01, regularization=0.02, n_epochs=20):
        self.n_users = n_users
        self.n_products = n_products
        self.K = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.epochs = n_epochs

        # Initialize Latent Factors (Gaussian distribution)
        scale = 1.0 / np.sqrt(self.K)
        self.P = np.random.normal(0, scale=scale, size=(self.n_users, self.K))
        self.Q = np.random.normal(0, scale=scale, size=(self.n_products, self.K))

        # Initialize Biases
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_products)
        self.global_mean = 0.0

        # History for plotting
        self.train_rmse_history = []
        self.test_rmse_history = []

    def fit(self, train_data, test_data=None, verbose=True):
        """
        Trains the model using SGD with Time Decay weighting.
        Args:
            train_data (np.ndarray): (N, 6) matrix. Col 5 is 'Time_Weight'.
            test_data (np.ndarray): Optional validation set.
            verbose (bool): Print logs per epoch.
        """
        print(f"Training MF Model (Factors={self.K}, LR={self.lr}, Reg={self.reg})...")

        self.global_mean = np.mean(train_data[:, 2])

        # Pre-fetch columns for speed
        users = train_data[:, 0].astype(int)
        products = train_data[:, 1].astype(int)
        ratings = train_data[:, 2]
        weights = train_data[:, 5]

        n_samples = train_data.shape[0]

        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Shuffle data indices for SGD
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            # SGD Loop
            for idx in indices:
                u, i_idx, r, w = users[idx], products[idx], ratings[idx], weights[idx]
                # Predict
                dot = np.dot(self.P[u], self.Q[i_idx])
                prediction = self.global_mean + self.b_u[u] + self.b_i[i_idx] + dot

                # Error
                error = r - prediction

                # Update with Time Weight
                weighted_error = error * w

                # Update Biases
                self.b_u[u] += self.lr * (weighted_error - self.reg * self.b_u[u])
                self.b_i[i_idx] += self.lr * (weighted_error - self.reg * self.b_i[i_idx])

                # Update Latent Factors
                p_u_current = self.P[u].copy()
                self.P[u] += self.lr * (weighted_error * self.Q[i_idx] - self.reg * p_u_current)
                self.Q[i_idx] += self.lr * (weighted_error * p_u_current - self.reg * self.Q[i_idx])

            # evaluate RMSE
            train_rmse = self.evaluate(train_data)
            self.train_rmse_history.append(train_rmse)

            log_msg = f"Epoch {epoch+1}/{self.epochs} - Train RMSE: {train_rmse:.4f}"

            if test_data is not None:
                test_rmse = self.evaluate(test_data)
                self.test_rmse_history.append(test_rmse)
                log_msg += f" - Test RMSE: {test_rmse:.4f}"
            log_msg += f" - Time: {time.time() - start_time:.2f}s"

            if verbose:
                print(log_msg)

    def evaluate(self, data):
        """Calculates RMSE."""
        true_ratings = data[:, 2]
        users = data[:, 0].astype(int)
        products = data[:, 1].astype(int)

        # Vectorized dot product: (N, K) * (N, K) -> sum axis 1 -> (N,)
        dot_products = np.sum(self.P[users] * self.Q[products], axis=1)

        predictions = self.global_mean + self.b_u[users] + self.b_i[products] + dot_products

        rmse = np.sqrt(np.mean((true_ratings - predictions) ** 2))
        return rmse
    
    def predict(self, u, i):
        """Predict score for a single user-item pair."""
        # Check boundary to avoid error if index is out of range
        if u >= self.n_users or i >= self.n_products:
            return self.global_mean
            
        pred = self.global_mean + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i])
        return pred
    
    def recommend(self, user_id, top_k=5):
        """Returns indices of top K recommended items for a user."""
        if user_id >= self.n_users:
            return []

        # Score all items: Global + UserBias + ItemBias + (UserFactor . ItemFactors.T)
        scores = self.global_mean + self.b_u[user_id] + self.b_i + np.dot(self.P[user_id], self.Q.T)
        top_prod_indices = np.argsort(scores)[-top_k:][::-1]
        return top_prod_indices