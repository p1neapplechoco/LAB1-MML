from itertools import chain, combinations
from sklearn.metrics import r2_score
import numpy as np

class FeatureSelection: 
    @staticmethod
    def __all_subsets(iterable, min_feat, max_feat):
        s = list(iterable)
        if max_feat is None:
            max_feat = len(s)
        return chain.from_iterable(combinations(s, r) for r in range(min_feat, max_feat + 1))

    @staticmethod
    def subset_selection(train, val, target, model, 
                         oneHotCols=None, min_feat=1, max_feat=None, max_diff=0.08):
        feature_columns = [col for col in train.select_dtypes(include='number').columns if col != target]
        
        one_hot_groups = {}
        if oneHotCols is None:
            oneHotCols = []
        for base in oneHotCols:
            group = [col for col in feature_columns if col.startswith(base + '_')]
            if group: 
                one_hot_groups[base] = group

        non_onehot_features = [col for col in feature_columns if all(not col.startswith(base + '_') for base in oneHotCols)]
        
        new_features = non_onehot_features + list(one_hot_groups.keys())
        
        best_adj_r2 = -float('inf')
        best_subset = None

        for subset in FeatureSelection.__all_subsets(new_features, min_feat, max_feat):
            subset = list(subset)
            
            final_subset = subset.copy()
            for base, group in one_hot_groups.items():
                if base in subset:
                    final_subset.extend(group)
                    final_subset.remove(base)
                    
            final_subset = list(set(final_subset))
            p = len(final_subset)
            if p == 0:
                continue

            train_df = train[final_subset + [target]]
            val_df = val[final_subset + [target]]

            X_train = train_df.drop(columns=target)
            y_train = train_df[target]
            X_val = val_df.drop(columns=target)
            y_val = val_df[target]

            n_train = len(X_train)
            n_val = len(X_val)

            if n_train - p - 1 <= 0 or n_val - p - 1 <= 0:
                continue

            model.fit(X_train, y_train)

            y_pred_val = model.predict(X_val)
            y_pred_train = model.predict(X_train)

            r2_val_base = r2_score(y_val, y_pred_val)
            r2_train_base = r2_score(y_train, y_pred_train)

            adj_r2_val = 1 - ((1 - r2_val_base) * (n_val - 1) / (n_val - p - 1))
            adj_r2_train = 1 - ((1 - r2_train_base) * (n_train - 1) / (n_train - p - 1))

            if adj_r2_val > best_adj_r2 and abs(adj_r2_val - adj_r2_train) < max_diff:
                best_adj_r2 = adj_r2_val
                best_subset = final_subset

        return best_subset, best_adj_r2
    
    @staticmethod
    def forward_selection_r2(df, target, model, oneHotCols=None):
        all_num_cols = df.select_dtypes(include='number').columns.tolist()
        features = [col for col in all_num_cols if col != target]
        
        # Nếu có chỉ định oneHotCols, nhóm các cột one hot lại với nhau
        one_hot_groups = {}
        if oneHotCols is not None:
            for base in oneHotCols:
                # Tìm tất cả các cột có định dạng base_<value>
                group_cols = [col for col in features if col.startswith(base + "_")]
                if group_cols:
                    one_hot_groups[base] = group_cols

        # Các cột không thuộc nhóm one hot (tức không nằm trong bất cứ nhóm nào)
        group_cols_all = set()
        for group in one_hot_groups.values():
            group_cols_all.update(group)
        remaining_individual = [col for col in features if col not in group_cols_all]
        
        # Các nhóm one hot
        remaining_groups = list(one_hot_groups.keys())
        
        best_subset = []
        best_adj_r2 = -float('inf')

        remaining_candidates = [('individual', feat) for feat in remaining_individual] + \
                            [('group', group) for group in remaining_groups]
        n = len(df)
        
        improvement = True
        while remaining_candidates and improvement:
            improvement = False
            best_candidate = None
            candidate_adj_r2 = best_adj_r2

            for cand_type, candidate in remaining_candidates:
                # Nếu là cột đơn, thêm trực tiếp; nếu là nhóm, thêm tất cả các cột trong nhóm đó
                if cand_type == 'individual':
                    current_features = best_subset + [candidate]
                elif cand_type == 'group':
                    current_features = best_subset + one_hot_groups[candidate]
                
                p = len(current_features)
                if n - p - 1 <= 0:
                    continue

                X = df[current_features]
                y = df[target]
                model.fit(X, y)
                y_pred = model.predict(X)
                r2_val = r2_score(y, y_pred)
                adj_r2 = 1 - ((1 - r2_val) * (n - 1) / (n - p - 1))
                
                if adj_r2 > candidate_adj_r2:
                    candidate_adj_r2 = adj_r2
                    best_candidate = (cand_type, candidate)
            
            if best_candidate is not None and candidate_adj_r2 > best_adj_r2:
                cand_type, candidate = best_candidate
                if cand_type == 'individual':
                    best_subset.append(candidate)
                else:
                    best_subset.extend(one_hot_groups[candidate])
                remaining_candidates = [item for item in remaining_candidates if not (item[0] == cand_type and item[1] == candidate)]
                best_adj_r2 = candidate_adj_r2
                improvement = True
            
        return best_subset, best_adj_r2

    @staticmethod
    def forward_selection_mse(df, target, model, oneHotCols=None):
        import numpy as np
        from sklearn.metrics import mean_squared_error

        # Lấy tất cả các cột số và loại bỏ cột target
        all_num_cols = df.select_dtypes(include='number').columns.tolist()
        features = [col for col in all_num_cols if col != target]
        
        # Nếu có chỉ định oneHotCols, nhóm các cột one hot lại với nhau
        one_hot_groups = {}
        if oneHotCols is not None:
            for base in oneHotCols:
                # Tìm tất cả các cột có định dạng base_<value>
                group_cols = [col for col in features if col.startswith(base + "_")]
                if group_cols:
                    one_hot_groups[base] = group_cols

        # Các cột không thuộc nhóm one hot (tức không nằm trong bất cứ nhóm nào)
        group_cols_all = set()
        for group in one_hot_groups.values():
            group_cols_all.update(group)
        remaining_individual = [col for col in features if col not in group_cols_all]
        
        # Các nhóm one hot
        remaining_groups = list(one_hot_groups.keys())
        
        best_subset = []
        best_mse = float('inf')  # Với MSE, càng nhỏ càng tốt

        remaining_candidates = [('individual', feat) for feat in remaining_individual] + \
                            [('group', group) for group in remaining_groups]
        n = len(df)
        
        improvement = True
        while remaining_candidates and improvement:
            improvement = False
            best_candidate = None
            candidate_mse = best_mse

            for cand_type, candidate in remaining_candidates:
                # Nếu là cột đơn, thêm trực tiếp; nếu là nhóm, thêm tất cả các cột trong nhóm đó
                if cand_type == 'individual':
                    current_features = best_subset + [candidate]
                elif cand_type == 'group':
                    current_features = best_subset + one_hot_groups[candidate]
                
                p = len(current_features)
                if n - p - 1 <= 0:
                    continue

                X = df[current_features]
                y = df[target]
                model.fit(X, y)
                y_pred = model.predict(X)
                mse_val = mean_squared_error(y, y_pred)
                
                if mse_val < candidate_mse:
                    candidate_mse = mse_val
                    best_candidate = (cand_type, candidate)
            
            if best_candidate is not None and candidate_mse < best_mse:
                cand_type, candidate = best_candidate
                if cand_type == 'individual':
                    best_subset.append(candidate)
                else:
                    best_subset.extend(one_hot_groups[candidate])
                # Loại bỏ ứng viên đã được thêm ra khỏi danh sách candidates
                remaining_candidates = [item for item in remaining_candidates if not (item[0] == cand_type and item[1] == candidate)]
                best_mse = candidate_mse
                improvement = True
            
        return best_subset, best_mse

    @staticmethod
    def forward_selection_mae(df, target, model, oneHotCols=None):
        from sklearn.metrics import mean_absolute_error

        all_num_cols = df.select_dtypes(include='number').columns.tolist()
        features = [col for col in all_num_cols if col != target]
        
        one_hot_groups = {}
        if oneHotCols is not None:
            for base in oneHotCols:
                group_cols = [col for col in features if col.startswith(base + "_")]
                if group_cols:
                    one_hot_groups[base] = group_cols

        group_cols_all = set()
        for group in one_hot_groups.values():
            group_cols_all.update(group)
        remaining_individual = [col for col in features if col not in group_cols_all]
        
        remaining_groups = list(one_hot_groups.keys())
        
        best_subset = []
        best_mae = float('inf') 

        remaining_candidates = [('individual', feat) for feat in remaining_individual] + \
                            [('group', group) for group in remaining_groups]
        n = len(df)
        
        improvement = True
        while remaining_candidates and improvement:
            improvement = False
            best_candidate = None
            candidate_mae = best_mae

            for cand_type, candidate in remaining_candidates:
                if cand_type == 'individual':
                    current_features = best_subset + [candidate]
                elif cand_type == 'group':
                    current_features = best_subset + one_hot_groups[candidate]
                
                p = len(current_features)
                if n - p - 1 <= 0:
                    continue

                X = df[current_features]
                y = df[target]
                model.fit(X, y)
                y_pred = model.predict(X)
                mae_val = mean_absolute_error(y, y_pred)
                
                if mae_val < candidate_mae:
                    candidate_mae = mae_val
                    best_candidate = (cand_type, candidate)
            
            if best_candidate is not None and candidate_mae < best_mae:
                cand_type, candidate = best_candidate
                if cand_type == 'individual':
                    best_subset.append(candidate)
                else:
                    best_subset.extend(one_hot_groups[candidate])

                remaining_candidates = [item for item in remaining_candidates if not (item[0] == cand_type and item[1] == candidate)]
                best_mae = candidate_mae
                improvement = True
            
        return best_subset, best_mae


