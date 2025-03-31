from itertools import chain, combinations
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

class FeatureSelection: 
    @staticmethod
    def forward_selection_adjr2(df, target, model, oneHotCols=None, log_transform=True):
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
                model.fit(X, y, log_transform=log_transform)
                y_pred = model.predict(X, y_exp = not log_transform)
                if log_transform:
                    y_pred = np.exp(y_pred)

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
    def forward_selection_mae(df, target, model, oneHotCols=None, log_transform=True):

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
                model.fit(X, y, log_transform=log_transform)
                y_pred = model.predict(X, y_exp = not log_transform)
                if log_transform:
                    y_pred = np.exp(y_pred)
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

    @staticmethod
    def backward_elimination_mae(df, target, model, oneHotCols=None, log_transform=True):

        # Lấy tất cả các cột số và loại bỏ cột target
        all_num_cols = df.select_dtypes(include='number').columns.tolist()
        features = [col for col in all_num_cols if col != target]
        
        # Xử lý nhóm one-hot nếu được chỉ định
        one_hot_groups = {}
        if oneHotCols is not None:
            for base in oneHotCols:
                group_cols = [col for col in features if col.startswith(base + "_")]
                if group_cols:
                    one_hot_groups[base] = group_cols

        # Xác định các cột không thuộc nhóm one-hot
        group_cols_all = set()
        for group in one_hot_groups.values():
            group_cols_all.update(group)
        individual_features = [col for col in features if col not in group_cols_all]
        
        # Khởi tạo tập các đặc trưng ban đầu: bao gồm các đặc trưng riêng lẻ và tất cả các đặc trưng của nhóm one-hot
        current_features = individual_features.copy()
        for group in one_hot_groups.values():
            current_features.extend(group)
        
        n = len(df)
        # Tính MAE ban đầu với toàn bộ các đặc trưng
        X = df[current_features]
        y = df[target]
        p = len(current_features)

        model.fit(X, y, log_transform=log_transform)
        y_pred = model.predict(X, y_exp=not log_transform)
        if log_transform:
            y_pred = np.exp(y_pred)
        best_mae = mean_absolute_error(y, y_pred)

        improvement = True
        # Vòng lặp loại bỏ đặc trưng cho đến khi không còn cải thiện MAE
        while improvement:
            improvement = False
            best_candidate = None
            candidate_mae = best_mae
            candidate_type = None  # 'individual' hoặc 'group'
            
            # Xét khả năng loại bỏ từng đặc trưng riêng lẻ (chỉ xét các cột không thuộc nhóm one-hot)
            for feat in [f for f in current_features if f not in group_cols_all]:
                new_features = [f for f in current_features if f != feat]
                if len(new_features) < 1:
                    continue
                X_new = df[new_features]
                p = len(new_features)
                if n - p - 1 <= 0:
                    continue
                model.fit(X_new, y, log_transform=log_transform)
                y_pred_new = model.predict(X_new, y_exp=not log_transform)
                if log_transform:
                    y_pred_new = np.exp(y_pred_new)
                mae_val = mean_absolute_error(y, y_pred_new)
                if mae_val < candidate_mae:
                    candidate_mae = mae_val
                    best_candidate = feat
                    candidate_type = 'individual'
            
            # Xét khả năng loại bỏ cả nhóm one-hot nếu toàn bộ các cột của nhóm hiện diện
            for base, group_cols in one_hot_groups.items():
                if all(col in current_features for col in group_cols):
                    new_features = [f for f in current_features if f not in group_cols]
                    if len(new_features) < 1:
                        continue
                    X_new = df[new_features]
                    p = len(new_features)
                    if n - p - 1 <= 0:
                        continue
                    model.fit(X_new, y, log_transform=log_transform)
                    y_pred_new = model.predict(X_new, y_exp=not log_transform)
                    if log_transform:
                        y_pred_new = np.exp(y_pred_new)
                    mae_val = mean_absolute_error(y, y_pred_new)
                    if mae_val < candidate_mae:
                        candidate_mae = mae_val
                        best_candidate = base
                        candidate_type = 'group'
            
            # Nếu tìm thấy candidate loại bỏ cải thiện MAE, cập nhật danh sách các đặc trưng hiện tại
            if best_candidate is not None and candidate_mae < best_mae:
                if candidate_type == 'individual':
                    current_features.remove(best_candidate)
                elif candidate_type == 'group':
                    for col in one_hot_groups[best_candidate]:
                        if col in current_features:
                            current_features.remove(col)
                best_mae = candidate_mae
                improvement = True

        return current_features, best_mae

    @staticmethod
    def backward_elimination_adjr2(df, target, model, oneHotCols=None, log_transform=True):

        # Lấy tất cả các cột số và loại bỏ cột target
        all_num_cols = df.select_dtypes(include='number').columns.tolist()
        features = [col for col in all_num_cols if col != target]
        
        # Nếu có chỉ định oneHotCols, gom các cột one hot lại với nhau
        one_hot_groups = {}
        if oneHotCols is not None:
            for base in oneHotCols:
                # Tìm các cột có dạng base_<value>
                group_cols = [col for col in features if col.startswith(base + "_")]
                if group_cols:
                    one_hot_groups[base] = group_cols

        # Các cột không thuộc bất kỳ nhóm one hot nào
        group_cols_all = set()
        for group in one_hot_groups.values():
            group_cols_all.update(group)
        remaining_individual = [col for col in features if col not in group_cols_all]
        
        # Các nhóm one hot
        remaining_groups = list(one_hot_groups.keys())
        
        # Khởi tạo best_subset gồm tất cả các cột: các biến đơn lẻ và tất cả các cột trong các nhóm one hot
        best_subset = []
        best_subset.extend(remaining_individual)
        for group in remaining_groups:
            best_subset.extend(one_hot_groups[group])
        
        # Tính Adjusted R² ban đầu với toàn bộ biến
        n = len(df)
        p = len(best_subset)
        if n - p - 1 <= 0:
            best_adj_r2 = -float('inf')
        else:
            X = df[best_subset]
            y = df[target]
            model.fit(X, y, log_transform=log_transform)
            y_pred = model.predict(X, y_exp=(not log_transform))
            if log_transform:
                y_pred = np.exp(y_pred)
            r2_val = r2_score(y, y_pred)
            best_adj_r2 = 1 - ((1 - r2_val) * (n - 1) / (n - p - 1))
        
        improvement = True
        # Vòng lặp loại bỏ từng biến hoặc nhóm nếu chỉ số Adjusted R² cải thiện
        while improvement:
            improvement = False
            best_candidate = None
            candidate_adj_r2 = best_adj_r2
            
            # Tạo danh sách các ứng viên loại bỏ từ tập hiện tại
            candidate_list = []
            # Các biến đơn lẻ
            for feat in remaining_individual:
                if feat in best_subset:
                    candidate_list.append(('individual', feat))
            # Các nhóm one hot (chỉ xét nếu tất cả các cột của nhóm có trong best_subset)
            for group, cols in one_hot_groups.items():
                if all(col in best_subset for col in cols):
                    candidate_list.append(('group', group))
            
            # Nếu không còn ứng viên nào, thoát vòng lặp
            if not candidate_list:
                break
            
            # Duyệt từng ứng viên loại bỏ và tính chỉ số Adjusted R² sau khi loại bỏ
            for cand_type, candidate in candidate_list:
                if cand_type == 'individual':
                    current_features = [feat for feat in best_subset if feat != candidate]
                elif cand_type == 'group':
                    current_features = [feat for feat in best_subset if feat not in one_hot_groups[candidate]]
                
                p_new = len(current_features)
                if n - p_new - 1 <= 0:
                    continue
                
                X = df[current_features]
                y = df[target]
                model.fit(X, y, log_transform=log_transform)
                y_pred = model.predict(X, y_exp=(not log_transform))
                if log_transform:
                    y_pred = np.exp(y_pred)
                r2_val = r2_score(y, y_pred)
                adj_r2 = 1 - ((1 - r2_val) * (n - 1) / (n - p_new - 1))
                
                # Nếu cải thiện Adjusted R², lưu lại ứng viên tốt nhất
                if adj_r2 > candidate_adj_r2:
                    candidate_adj_r2 = adj_r2
                    best_candidate = (cand_type, candidate)
            
            # Nếu có ứng viên tốt nhất làm cải thiện Adjusted R², cập nhật best_subset
            if best_candidate is not None and candidate_adj_r2 > best_adj_r2:
                cand_type, candidate = best_candidate
                if cand_type == 'individual':
                    best_subset = [feat for feat in best_subset if feat != candidate]
                else:  # loại bỏ cả nhóm
                    best_subset = [feat for feat in best_subset if feat not in one_hot_groups[candidate]]
                best_adj_r2 = candidate_adj_r2
                improvement = True
                
        return best_subset, best_adj_r2
