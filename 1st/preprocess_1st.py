import pandas as pd

def preporcessing(train_data, test_data) :

    #eva air in temp_MAX	eva air out temp_MIN 에 over 결측치 존재. 결측치가 존재하는 행만 삭제
    train_over = train_data[(train_data['eva air out temp_MAX']=='+OVER') | (train_data['eva air out temp_MAX']=='-OVER')].index
    train_data.drop(train_over, inplace=True)
    train_over2 = train_data[(train_data['eva air out temp_MIN']=='+OVER') | (train_data['eva air out temp_MIN']=='-OVER')].index
    train_data.drop(train_over2, inplace=True)

    test_over = test_data[(test_data['eva air out temp_MAX']=='+OVER') | (test_data['eva air out temp_MAX']=='-OVER')].index
    test_data.drop(test_over, inplace=True)
    test_over2 = test_data[(test_data['eva air out temp_MIN']=='+OVER') | (test_data['eva air out temp_MIN']=='-OVER')].index
    test_data.drop(test_over2, inplace=True)
    test_over3 = test_data[(test_data['outside temp_MAX']=='+OVER')].index
    test_data.drop(test_over3, inplace=True)


    train_data_X = train_data.iloc[:, :-1]
    train_data_y = train_data.iloc[:, -1]

    test_data_X = test_data.iloc[:, :-1]
    test_data_y = test_data.iloc[:, -1]

    return train_data_X, train_data_y, test_data_X,  test_data_y