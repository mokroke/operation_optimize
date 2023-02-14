import pandas as pd
from sklearn import preprocessing
import numpy as np
import pickle


class Numerical_Inversion():
    def __init__(self, df, not_use_col_list):
        self.df = df
        self.not_use_col_list = not_use_col_list
        
        #dataframeを変換するものとしないもので分ける
        self.convert_df = df.drop(not_use_col_list, axis=1)
        self.not_convert_df = df[not_use_col_list]
        
        
    def convert(self, option):
        
        #'SS'は標準化
        if option == 'SS':
            df_converted = self.standard_scale(self.convert_df)
            return df_converted
        
        #'MM'は正規化
        if option == 'MM':
            df_converted = self.minmax_scale(self.convert_df)
            return df_converted
        
        #'RS'はロバスト化
        if option == 'RS':
            df_converted = self.robust_scale(self.convert_df)
            return df_converted
        
        #'LOG'は対数変換
        if option == 'LOG':
            df_converted = self.log_scale(self.convert_df)
            return df_converted
        
        #'YEO'はyeo-johnson変換
        if option == 'YEO':
            df_converted = self.yeo_scale(self.convert_df)
            return df_converted
        
        print('対応するコマンドを入力してください')
        print('["SS", "MM", "RS", "LOG", "YEO"]')
        
    def make_converted_df(self, option):
        df_converted = self.convert(option)
        all_df = pd.concat([df_converted, self.not_convert_df], axis=1)
        return all_df
        
    def yeo_scale(self, df):
        from sklearn.preprocessing import MinMaxScaler, PowerTransformer
        mm = MinMaxScaler()
        pt = PowerTransformer(standardize=False)
        df_mm = mm.fit_transform(df)
        df_pt_row = pt.fit_transform(df_mm)
        df_pt = pd.DataFrame(df_pt_row, columns=df.columns)
        return df_pt
    
    def log_scale(self, df):
        import numpy as np
        df_log_row = df.apply(np.log1p)
        df_log = pd.DataFrame(df_log_row, columns=df.columns)
        return df_log
    
    def robust_scale(self, df):
        from sklearn.preprocessing import RobustScaler
        rs = RobustScaler()
        df_rs_row = rs.fit_transform(df)
        df_rs = pd.DataFrame(df_rs_row, columns=df.columns)
        return df_rs
    
    
    def standard_scale(self, df):
        from sklearn.preprocessing import StandardScaler
        ss = StandardScaler()
        df_ss_row = ss.fit_transform(df)
        df_ss = pd.DataFrame(df_ss_row, columns=df.columns)
        return df_ss
    
    def minmax_scale(self, df):
        from sklearn.preprocessing import MinMaxScaler
        mm = MinMaxScaler()
        df_mm_row = mm.fit_transform(df)
        df_mm = pd.DataFrame(df_mm_row, columns=df.columns)
        return df_mm
    
    
    
from datetime import date
from dateutil.relativedelta import relativedelta
import lightgbm as lgb
class make_tr_va_te():
    def __init__(self, df, train_end_next_date):
        self.df = df
        self.train_end_next_date = train_end_next_date
        if type(df['date'][0]):
            df['date'] = pd.to_datetime(df['date'])
        
    def make_train_data(self):
        train_all = self.df[self.df['predict'] == 0]
        train = train_all[train_all['date'] < self.train_end_next_date]
        train_notna = train[train['bikes_available'].notna()]
        return train_notna
        
    def make_valid_data(self):
        valid_all = self.df[self.df['predict'] == 0]
        valid = valid_all[(self.train_end_next_date <= valid_all['date']) & (valid_all['date'] < (self.train_end_next_date + relativedelta(months = 1)))]
        valid_notna = valid[valid['bikes_available'].notna()]
        return valid_notna
                                      
    def make_test_data(self):
        test_all = self.df[self.df['predict'] == 1]
        test = test_all[((self.train_end_next_date + relativedelta(months = 1)) <= test_all['date']) & (test_all['date']< (self.train_end_next_date + relativedelta(months = 2)))]
        return test
    
    def model_for_data(self, train, valid):
        tr_X = train.drop(['id','predict','bikes_available','date'],axis=1)
        tr_y = train['bikes_available']
        va_X = valid.drop(['id','predict','bikes_available','date'],axis=1)
        va_y = valid['bikes_available']   
        return tr_X, tr_y, va_X, va_y
    
    def predict_for_data(self, test):
        te_X = test.drop(['id','predict','bikes_available','date'],axis=1)       
        return te_X
    
    def make_fit_model(self, tr_X, tr_y, va_X, va_y, params):
        lgb_train = lgb.Dataset(tr_X, tr_y)
        lgb_eval = lgb.Dataset(va_X, va_y)
#         params = {
#             'task':'train',
#             'boosting_type':'gbdt',
#             'objective':'regression',
#             'metric':{'rmse'},
#             'learning_rate':0.01,
#             'num_leaves':23,
#             'min_data_in_leaf':1,
#             'num_iteration': 10000,
#             'verbose':0,
#             'random_seed':0,
#         }

        model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval],
                         verbose_eval=100, num_boost_round=10000,
                         early_stopping_rounds=100)
        return model
    
    def predict(self,params,model_path ,model_file_name):
        import pickle
        
        train = self.make_train_data()
        valid = self.make_valid_data()
        test = self.make_test_data()
        tr_X, tr_y, va_X, va_y = self.model_for_data(train, valid)
        te_X = self.predict_for_data(test)
        model  = self.make_fit_model(tr_X, tr_y, va_X, va_y, params)
        pickle.dump(model, open(model_path + model_file_name, 'wb'))
        valid_best_score = model.best_score['valid_1']['rmse']
        y_pred = model.predict(te_X)
        sub_index = test['id']
        sub_df = pd.DataFrame(list(zip(sub_index, y_pred)))
        print('*****')
        print(valid_best_score)
        return sub_df,valid_best_score
    
    
def month_range(start, stop, step = relativedelta(months = 1)):
    current = start
    while current < stop:
        yield current
        current += step