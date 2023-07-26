# FC_simple

get_predict function takes (df: pd.DataFrame,
                start: str,
                end: str,
                frequency: str,
                cols_fcst: List[str],
                cols_param: List[str],
                cols_group: List[str])
and return daframe with two additional columns: name of model and forecast

example
    test_df = get_sql.get_df('frct_data.sql')
    predict = get_predict(df= test_df,
                          start= pd.to_datetime('today'),
                          end= '2024-12-31',
                          frequency= 'week',
                          cols_fcst= ['pcs_shipped'],
                          cols_param= ['even_week','year_iso','week'],
                          cols_group= ['ID_obj', 'item_group'])

    predict.to_excel('predict.xlsx')


input data (test_df):
ID_obj	year_iso	week	periods	even_week	item_group	pcs_shipped	pcs_shipped_PROMO
186	2020		1	0	0		-1	5	67674		0

output data:
ID_obj	year_iso	week	periods	even_week	item_group	pcs_shipped	pcs_shipped_PROMO	mod_pcs_shipped		fc_pcs_shipped
186	2020		1	0	0		-1	5	67674		0			LinearRegression	50235
