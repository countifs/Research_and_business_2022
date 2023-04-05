# Research_and_business_2022

## Intro

이 프로젝트는 2022년도 산학협력 프로젝트에서 진행하였던 분석을 따로 정리한 것 입니다.
주어진 데이터셋은 회사에서 제공받은 데이터셋으로, 냉장고에 부착되어있는 센서들의 값들이 기록되어있습니다. 이 산학협력 프로젝트의 최종 목표는 주어진 센서들의 값들을 활용하여 냉장고에 적상 발생유무를 판단하는 것이었습니다. 적상이 있음, 없음에 대한 분류를 진행하기 위해, 중요도가 높은 센서들을 특정하고자 하였습니다.

## 1st, 2nd 폴더
두 폴더내에 있는 코드들은 비슷한 방식의 전처리 및 분석 방법을 구현한 것입니다. 
1st 와 2nd 폴더로 나눈 이유는 다룬 데이터셋의 내용 및 feature의 종류가 달라 부득이하게 분리해놓았습니다. 

### code 설명
각 폴더내의 preprocess_1st.py, preprocess_2nd.py는 pandas를 사용하여 None 값을 가지는 행의 값들을 drop 하고, 분류해야할 라벨이 있는 y와 입력데이터 X로 나누는 코드입니다.
~_analysis_RF.py와 ~_analysis_SVC.py 는 skit-learn 라이브러리를 사용하여 random forest와 SVC를 불러온 다음, 분류를 진행한 코드입니다. 추가적으로 분류를 위한 최적의 feature를 찾기 위해 RFECV를 사용하여 최적의 feature갯수를 구했습니다. 또한, 회사측에서 중요하다고 생각한 feature들만을 사용하였을 때의 성능확인도 진행하였습니다.


## 환경설정
해당 코드를 실행하기 위해서는 pandas, numpy, matplotlib, skit-learn 라이브러리가 필요합니다.
다음의 커맨드를 사용하여 다운 받을 수 있습니다.
```
pip install pandas scikit-learn numpy matplotlib 
```
