import numpy as np
from scipy.linalg import svd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class L2NormalizedPCA(BaseEstimator, TransformerMixin):
    """
    基于L2归一化的PCA实现
    
    该方法首先对每个样本进行L2归一化，然后执行SVD分解，
    可选地在降维后恢复样本的原始长度。
    
    参数:
    n_components (int): 要保留的主成分数量
    restore_norms (bool): 是否在降维后恢复样本的原始L2范数
    copy (bool): 是否复制输入数据
    """
    
    def __init__(self, n_components=None, restore_norms=True, copy=True):
        self.n_components = n_components
        self.restore_norms = restore_norms
        self.copy = copy
        
    def fit(self, X, y=None):
        """
        拟合模型到数据
        
        参数:
        X (array-like): 形状为(n_samples, n_features)的输入数据
        y: 忽略，为了API一致性
        
        返回:
        self: 拟合后的模型实例
        """
        # 验证输入
        X = check_array(X, copy=self.copy)
        n_samples, n_features = X.shape
        
        # 确定要保留的主成分数量
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        else:
            n_components = self.n_components
            
        if n_components > min(n_samples, n_features):
            raise ValueError("n_components cannot be larger than min(n_samples, n_features)")
        
        self.n_components_ = n_components
        
        # 1. 计算并存储原始样本的L2范数
        self.original_norms_ = np.linalg.norm(X, axis=1)
        
        # 避免除零错误（对于零向量，范数设为1，归一化后仍为零向量）
        zero_norm_mask = self.original_norms_ == 0
        self.original_norms_[zero_norm_mask] = 1.0
        
        # 2. 对数据进行L2归一化
        X_normalized = X / self.original_norms_[:, np.newaxis]
        
        # 3. 对归一化后的数据执行SVD
        # 注意：这里不使用中心化，与标准PCA不同
        U, S, Vt = svd(X_normalized, full_matrices=False)
        
        # 存储主成分（右奇异向量）
        self.components_ = Vt[:n_components].T
        
        # 存储奇异值
        self.singular_values_ = S[:n_components]
        
        # 计算解释方差比
        explained_variance = (S ** 2) / (n_samples - 1)
        total_var = explained_variance.sum()
        self.explained_variance_ratio_ = explained_variance[:n_components] / total_var
        
        return self
        
    def transform(self, X):
        """
        将数据转换到主成分空间
        
        参数:
        X (array-like): 形状为(n_samples, n_features)的输入数据
        
        返回:
        X_transformed: 降维后的数据
        """
        # 检查是否已拟合
        check_is_fitted(self, ['components_', 'original_norms_'])
        
        # 验证输入
        X = check_array(X, copy=self.copy)
        
        # 1. 计算输入数据的L2范数
        input_norms = np.linalg.norm(X, axis=1)
        zero_norm_mask = input_norms == 0
        input_norms[zero_norm_mask] = 1.0  # 避免除零错误
        
        # 2. 对输入数据进行L2归一化
        X_normalized = X / input_norms[:, np.newaxis]
        
        # 3. 投影到主成分空间
        X_transformed = np.dot(X_normalized, self.components_)
        
        # 4. 如果需要，恢复原始长度
        if self.restore_norms:
            X_transformed = X_transformed * input_norms[:, np.newaxis]
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        拟合模型并转换数据
        
        参数:
        X (array-like): 形状为(n_samples, n_features)的输入数据
        y: 忽略，为了API一致性
        
        返回:
        X_transformed: 降维后的数据
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """
        将降维后的数据转换回原始空间
        
        参数:
        X (array-like): 形状为(n_samples, n_components)的降维数据
        
        返回:
        X_original: 重建的原始数据
        """
        # 检查是否已拟合
        check_is_fitted(self, ['components_'])
        
        # 验证输入
        X = check_array(X, copy=self.copy)
        
        # 1. 反转投影
        X_reconstructed = np.dot(X, self.components_.T)
        
        # 2. 如果转换时恢复了范数，这里需要先去除范数影响
        # 注意：这是一个近似，因为我们无法知道原始数据的精确范数
        # 对于新数据，我们无法准确恢复原始范数信息
        if self.restore_norms:
            # 这是一个启发式方法：使用训练数据的平均范数
            avg_norm = np.mean(self.original_norms_)
            X_reconstructed = X_reconstructed / avg_norm
        
        return X_reconstructed
    