# Research_and_business_2022

## Intro

이 프로젝트는 2022년도 산학협력 프로젝트에서 진행하였던 분석을 따로 정리한 것 입니다.
주어진 데이터셋은 회사에서 제공받은 데이터셋으로, 냉장고에 부착되어있는 센서들의 값들이 기록되어있습니다. 이 산학협력 프로젝트의 최종 목표는 주어진 센서들의 값들을 활용하여 냉장고에 적상 발생유무를 판단하는 것이었습니다. 적상이 있음, 없음에 대한 분류를 진행하기 위해, 중요도가 높은 센서들을 특정하고자 하였습니다.

##1st_analysis.ipynb, 2nd_analysis.py
해당 파일은 회사에서 제공한 데이터셋을 기반으로 None 값이 포함되어있는 행을 삭제하는 전처리를 진행하였습니다.
또한, Random forest 와 SVC을 사용하여 센서들의 분류를 진행하고, RFECV 를 사용하여 주어진 센서(입력된 feature들)의 조합을 바꾸어 가며 최적의 feature를 찾고자 하였습니다. 

## 
해당 코드를 실행하기 위해서는 pandas, numpy, matplotlib, skit-learn 라이브러리가 필요합니다.
다음의 커맨드를 사용하여 다운 받을 수 있습니다.
```
pip install pandas scikit-learn numpy matplotlib 
```
