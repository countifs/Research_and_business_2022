import pandas as pd

def preprocessing_2(train_data, test_data) :

    train_label = train_data.loc[:,'Heat_ON']
    train_set = list(set(train_label))



    from tqdm import tqdm

    train_datas = train_data.iloc[:]
    print( train_datas.iloc[1].index)

    for each_index in ['T1_lp1_MIN', 'T1_lp1_MAX', 'T2_hp1_MIN', 'T2_hp1_MAX', 'T3_hp2_MIN',
        'T3_hp2_MAX', 'T1_comp in_MIN', 'T1_comp in_MAX', 'T2_comp out_MIN',
        'T2_comp out_MAX', 'T3_cond in_MIN', 'T3_cond in_MAX',
        'T4_cond out_MIN', 'T4_cond out_MAX', 'T5_Exp in_MIN', 'T5_Exp in_MAX',
        'T6_Exp out_MIN', 'T6_Exp out_MAX', 'T7_evap in_MIN', 'T7_evap in_MAX',
        'T8_evap out1_MIN', 'T8_evap out1_MAX', 'T8_evap out2_MIN',
        'T8_evap out2_MAX', 'T10_sol out_MIN', 'T10_sol out_MAX',
        'inside temp_MIN', 'inside temp_MAX', 'center_MIN', 'center_MAX',
        'outside temp_MIN', 'outside temp_MAX', 'eva air in temp_MIN',
        'eva air in temp_MAX', 'eva air out temp_MIN', 'eva air out temp_MAX',
        'cond air in temp_MIN', 'cond air in temp_MAX', 'cond air out temp_MIN',
        'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'] :
        print(each_index)
        data_over = train_data[(train_data[each_index] == '+OVER')].index
        train_data.drop(data_over, inplace=True)
        data_over2 = train_data[(train_data[each_index] == '-OVER')].index
        train_data.drop(data_over2, inplace=True)

    for each_index2 in ['T1_lp1_MIN', 'T1_lp1_MAX', 'T2_hp1_MIN', 'T2_hp1_MAX', 'T3_hp2_MIN',
        'T3_hp2_MAX', 'T1_comp in_MIN', 'T1_comp in_MAX', 'T2_comp out_MIN',
        'T2_comp out_MAX', 'T3_cond in_MIN', 'T3_cond in_MAX',
        'T4_cond out_MIN', 'T4_cond out_MAX', 'T5_Exp in_MIN', 'T5_Exp in_MAX',
        'T6_Exp out_MIN', 'T6_Exp out_MAX', 'T7_evap in_MIN', 'T7_evap in_MAX',
        'T8_evap out1_MIN', 'T8_evap out1_MAX', 'T8_evap out2_MIN',
        'T8_evap out2_MAX', 'T10_sol out_MIN', 'T10_sol out_MAX',
        'inside temp_MIN', 'inside temp_MAX', 'center_MIN', 'center_MAX',
        'outside temp_MIN', 'outside temp_MAX', 'eva air in temp_MIN',
        'eva air in temp_MAX', 'eva air out temp_MIN', 'eva air out temp_MAX',
        'cond air in temp_MIN', 'cond air in temp_MAX', 'cond air out temp_MIN',
        'T8_1-T7_MIN', 'T8_1-T7_MAX', 'T8_2-T7_MIN', 'T8_2-T7_MAX'] :
        
        data_over3 = test_data[(test_data[each_index2] == '+OVER')].index
        test_data.drop(data_over3, inplace=True)
        data_over4 = test_data[(test_data[each_index2] == '-OVER')].index
        test_data.drop(data_over4, inplace=True)

            


    train_label = train_data.loc[:,'Heat_ON']
    train_set = list(set(train_label))



    train_label_60 = train_data.loc[(train_data['Heat_ON'] == 'Low(60%)')]
    train_label_70 = train_data.loc[(train_data['Heat_ON'] == 'Low(70%)')]
    train_label_80 = train_data.loc[(train_data['Heat_ON'] == 'Low(80%)')]
    train_label_90 = train_data.loc[(train_data['Heat_ON'] == 'Low(90%)')]
    train_label_on = train_data.loc[(train_data['Heat_ON'] == 'On')]
    train_label_off = train_data.loc[(train_data['Heat_ON'] == 'Off')]




    train_lablel_60 = train_label_60.sample(n=404)
    train_lablel_70 = train_label_70.sample(n=404)
    train_lablel_80 = train_label_80.sample(n=404)
    train_lablel_on = train_label_on.sample(n=404)
    train_lablel_off = train_label_off.sample(n=404)





    sampled_train_data = pd.concat([train_lablel_60, train_lablel_70, train_lablel_80, train_label_90, train_lablel_on, train_lablel_off])

    train_data_X = sampled_train_data.iloc[:, :-1]
    train_data_y = sampled_train_data.iloc[:, -1]

    test_data_X = test_data.iloc[:, :-1]
    test_data_y = test_data.iloc[:, -1]

    return train_data_X, train_data_y, test_data_X, test_data_y