''' simple regrassion predict to dataframe '''
import pandas as pd
from typing import List, Dict, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_absolute_error

def correlation(data: pd.DataFrame,
                base_column: str,
                except_columns: List[str] = None) -> List[Tuple[str, float]]:
    ''' correlation between base_column and all columns '''
    if except_columns is None:
        except_columns = [base_column]
    else:
        except_columns.append(base_column)
    crl = data.corr()[base_column]
    return sorted({k: v for k, v in crl.items() if k not in except_columns}.items(),
                  key=lambda x: abs(x[1]),
                  reverse=True)

def get_calendar(date_start: str,
                 date_end: str,
                 freq: str,
                 max_input: int) -> pd.DataFrame:
    ''' freq week or month '''
    if isinstance(date_start, str):
        date_start = datetime.strptime(date_start,'%Y-%m-%d')
    if isinstance(date_end, str):
        date_end = datetime.strptime(date_end,'%Y-%m-%d')
    assert freq in ['week', 'month'], 'Need to choose frequency week or month'
    if freq == 'week':
        offset = pd.Timedelta(date_start.weekday(), unit='D')
        pred_periods = pd.date_range(start=date_start.date() - offset,
                                     end=date_end.date(), freq='W-MON')
        return pd.DataFrame(list(zip([_.to_pydatetime().isocalendar().year for _ in pred_periods],
                                     [_.to_pydatetime().isocalendar().week for _ in pred_periods],
                                     [_ + 1 + max_input for _ in range(len(pred_periods))],
                                     [_.to_pydatetime().isocalendar().week%2 for _ in pred_periods]))
                            , columns=['year_iso', 'week', 'periods', 'even_week'])
    if freq == 'month':
        offset = pd.Timedelta(date_start.day + 1, unit='D')
        pred_periods = pd.date_range(start=date_start.date() - offset,
                                     end=date_end.date(), freq='MS')
        return pd.DataFrame(list(zip([_.to_pydatetime().year for _ in pred_periods],
                                     [_ + 1 + max_input for _ in range(len(pred_periods))],
                                     [_.to_pydatetime().month for _ in pred_periods]))
                            , columns=['year', 'periods', 'month'])

def models() -> Dict[str, object]:
    ''' all regression models to use '''
    all_m = [
        LinearRegression,
        KNeighborsRegressor,
        RandomForestRegressor
        ]
    return {_.__name__: _ for _ in all_m}

def compare_err(iterable: Dict[str, List[float]], facts: pd.Series) -> str:
    ''' best model by mae'''
    best_mod = None
    min_err = float('inf')
    for mod in iterable:
        error = mean_absolute_error(facts, iterable[mod])
        if min_err > error:
            min_err = error
            best_mod = mod
    return best_mod

def choose_best(y: pd.Series,
                x: pd.DataFrame,
                x_pred: pd.DataFrame,
                positive_only: bool = True)->List[List[str, List[float]]]:
    ''' return best of models name and result '''
    preds = {}
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    mods = models()
    for _name, _mod in mods.items():
        model = _mod()
        lof = LocalOutlierFactor()
        lof.fit(pd.DataFrame(data=y_train))
        outliers = lof.fit_predict (pd.DataFrame(data=y_train))
        X_train_clean, y_train_clean = X_train[outliers==1], y_train[outliers==1]
        model.fit(X_train_clean, y_train_clean)

        preds[_mod.__name__] = [0 if _ < 0 else _ for _ in model.predict(X_test)] \
            if positive_only \
            else model.predict(X_test)
    name_of_best = compare_err(preds, y_test)
    model = mods[name_of_best]()
    model.fit(x, y)
    result = model.predict(pd.concat([x, x_pred], ignore_index=False))
    result = [0 if _ < 0 else _ for _ in result] if positive_only else result
    return [[name_of_best, _] for _ in result]

def get_predict(df: pd.DataFrame,
                start: str,
                end: str,
                frequency: str,
                cols_fcst: List[str],
                cols_param: List[str],
                cols_group: List[str]) -> pd.DataFrame:
    ''' return dataframe with choosen predict'''
    # add columns with zero where is no data
    df = df.pivot(index=cols_param+['periods'],
                  columns=cols_group,
                  values=cols_fcst).fillna(0).stack(level=list(range(1, 1+len(cols_group)))).reset_index()
    df['input'] = True
    fcst_periods = get_calendar(date_start= start,
                                date_end= end,
                                freq= frequency,
                                max_input=max(df['periods']))
    fcst_cols = df.groupby(cols_group, group_keys=False)[cols_group].apply(lambda x: 1).reset_index()[cols_group]
    fcst_cols['input']= False
    for _ in cols_fcst:
        fcst_cols[_] = None
    fcst_cols = fcst_periods.merge(fcst_cols, how='cross')

    df = pd.concat([df, fcst_cols], ignore_index= True)
    df = df.sort_values(cols_group + list(fcst_periods.columns),
                        ascending=[True]*(len(cols_group)+fcst_periods.shape[1])).reset_index(drop=True)

    for _col_fcst in cols_fcst:
        df[['mod_'+_col_fcst,
            'fc_'+_col_fcst]] = pd.DataFrame(df.groupby(cols_group,
                                                        group_keys=False).apply(lambda sub_df:
                                                                            sub_df[sub_df['input']==True][[_col_fcst]]
                                                                            .apply(choose_best,
                                                                                   x = sub_df[sub_df['input']==True][cols_param],
                                                                                   x_pred = sub_df[sub_df['input']==False][cols_param]
                                                                                   )).iloc[:, 0].tolist())
    return df