# (주)제이오텍 의료용 냉장고의 센서 데이터 기반 이상탐지 프로젝트

## Intro

이 프로젝트는 2022년도 산학협력 프로젝트에서 진행하였던 분석을 따로 정리한 것 입니다.
주어진 데이터셋은 (주)제이오텍 에서 제공받은 데이터셋으로, 의료용 냉장고에 부착되어있는 센서들의 값들이 기록되어있습니다. 이 산학협력 프로젝트의 최종 목표는 주어진 센서들의 값들을 활용하여 의료용 냉장고에 적상 발생유무를 판단하는 것이었습니다. 적상이 있음, 없음에 대한 분류를 진행하기 위해, 중요도가 높은 센서들을 특정하고자 하였습니다. 제공되는 구글드라이브 내에 코드가 존재하며, '김재호' 폴더 내에 분석을 위해 작성한 코드가 업로드 되어있습니다. 
[산학협력_2022](https://drive.google.com/drive/folders/16lO8i1p6F5wzyI3iiVYBizSwwf3HI0St?usp=share_link)

<br>

# 2022 산학협력 프로젝트
팀원 - 김재호, 서승원, 박순혁, 윤하영 

<br>

# 역할 소개 (김재호)
- 프로젝트 기획 및 분석 방향 제시 (파일럿 테스트 수행)
- 데이터 클렌징, 전처리 역할 수행
- 데이터 시각화 파트 담당
  - 시계열 데이터 파트 : Power BI 활용
  - 센서별 변수 비교 : KNIME - plotly Rader 차트 활용
  - 센서별 Box plot 비교 : Python seaborn활용

![전처리 파트](https://github.com/countifs/Research_and_business_2022/blob/main/imgaes/%EC%82%B0%ED%95%99%ED%98%91%EB%A0%A5-%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%84%EC%B2%98%EB%A6%AC(%EA%B9%80%EC%9E%AC%ED%98%B8).png)

![시각화 파트](https://github.com/countifs/Research_and_business_2022/blob/main/imgaes/%EC%82%B0%ED%95%99%ED%98%91%EB%A0%A5-%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%8B%9C%EA%B0%81%ED%99%94(%EA%B9%80%EC%9E%AC%ED%98%B8).png)

![프로젝트 기획 및 파일럿 테스트 진행](https://github.com/countifs/Research_and_business_2022/blob/main/imgaes/%EC%82%B0%ED%95%99%ED%98%91%EB%A0%A5%20-%20%EB%B6%84%EC%84%9D%EA%B8%B0%ED%9A%8D%20%EB%B0%8F%20%ED%8C%8C%EC%9D%BC%EB%9F%BF%20%ED%85%8C%EC%8A%A4%ED%8A%B8(%EA%B9%80%EC%9E%AC%ED%98%B8).png)

![데이터 시각화 with KNIME](https://github.com/countifs/Research_and_business_2022/blob/main/imgaes/%EC%82%B0%ED%95%99%ED%98%91%EB%A0%A5%20-%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%8B%9C%EA%B0%81%ED%99%94%20with%20KNIME(%EA%B9%80%EC%9E%AC%ED%98%B8).png)

![데이터 시각화 예시 with KNIME](https://github.com/countifs/Research_and_business_2022/blob/main/imgaes/%EC%82%B0%ED%95%99%ED%98%91%EB%A0%A5%20-%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%8B%9C%EA%B0%81%ED%99%94%20%EC%98%88%EC%8B%9C%20with%20KNIME(%EA%B9%80%EC%9E%AC%ED%98%B8).png)

![데이터 시각화 예시 with PowerBi](https://github.com/countifs/Research_and_business_2022/blob/main/imgaes/%EC%82%B0%ED%95%99%ED%98%91%EB%A0%A5%20-%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%8B%9C%EA%B0%81%ED%99%94%20%EC%98%88%EC%8B%9C%20with%20PowerBi%20(%EA%B9%80%EC%9E%AC%ED%98%B8).png)

![데이터 시각화 예시 with Python](https://github.com/countifs/Research_and_business_2022/blob/main/imgaes/%EC%82%B0%ED%95%99%ED%98%91%EB%A0%A5%20-%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%8B%9C%EA%B0%81%ED%99%94%20%EC%98%88%EC%8B%9C%20with%20Python%20(%EA%B9%80%EC%9E%AC%ED%98%B8).png)

<br>

---

<br>

## 목표

### 1. 주어진 냉장고 센서 데이터셋을 사용하여 냉장고에 적상현상 발생을 감지하는 것
### 2. 추가적으로 보유하고 있는 냉장고 센서 데이터셋을 같이 활용하여 적상 및 냉매부족 현상 감지(냉매의 남은 용량을 10% 단위로 분류)

1. 주어진 데이터셋에 존재하는 feature 들의 중요도 계산 
- 센서 feature가 많기 때문에 그 중에서 적상현상 발생 유무, 냉매 용량 상황과 큰 상관관계를 갖는 feature를 찾기 위함

2. 주어진 데이터셋의 feature 중요도에 상관없이 적상 유무 분류에 유리한 feature의 갯수 계산 
- 적상 유무 및 냉매부족 현상의 분류를 위해 모든 feature를 사용하는 것 보다 중요한 feature들만 사용하여 효율적인 분류를 하기 위함
- 1번과 합쳐 중요도가 높은 센서 feature가 2번 feature들 목록에 포함된다면 적은 센서를 사용하여 더욱 효율적으로 냉매 부족 및 적상 유무를 분류한 수 있을 것임


3. 제공된 센서 feature 중 다른 위치에 부착된 같은 종류의 센서 2 개 feature 중 냉장고 적상 유무 및 냉매부족 상황 분류에 유리한 센서 분석
- 부착된 위치에 따라 냉장고의 적상과 냉매 상황을 더 잘 반영할 수 있는 위치를 찾기 위함



==> 중요도가 높은 feature를 포함하여 냉장고의 상황을 잘 분류 할 수 있는 feature들을 선택하고, 냉장고 상황을 더 잘 반영할 수 있는 센서의 위치를 유추하여 효율적으로 적상 유무와 냉매부족 분류를 진행.


## DATA
### 데이터 형태 
- 1 번 데이터 : 총 44개의 센서 종류(x) 및 적상 유무 라벨 1(y)
- 2 번 데이터 : 총 44개의 센서 종류(x) 및 적상 유무 + 냉매 부족 라벨 1(y)

### 데이터 preprocessing
(회사에서 제공한 데이터셋은 비공개이므로 센서 feature의 이름을 임의의 단어로 치환함
1. 모든 데이터는 exel 형태로 제공됨 
2. 훈련 및 테스트를 위해 전체 데이터셋을 7 : 3 (train : test) 으로 분할 
3. 결측치 전처리
- (1번 데이터) 총 5개의 센서 feature에서 결측치 제거
<br/> pandas dataframe 으로 데이터를 읽어온 다음, 조건문을 사용하여 2 종류의 결측치 탐지
<br/> .index 함수를 사용하여 결측치가 포함된 행 인덱스 추출 및 drop 함수를 사용하여 행 삭제 
- (2번 데이터) 
<br/> 1번 데이터보다 더 많은 데이터 양으로 인해 결측치가 존재하는 feature를 직접 찾기 어려워 for 문과 dataframe 조건문을 이용하여 결측치 탐지
<br/> 결측치가 탐지된 행의 번호를 .index 를 사용하여 추출 및 drop 함수를 사용하여 행 삭제
4. pandas의 iloc() 함수를 사용하여 target class 열 추출 
- 1번 데이터 : 적상 유무
- 2번 데이터 : 적상 유무 + 냉매 용량

### feature 중요도 계산 
1. scikit-learn 라이브러리의 random forest와 SVC를 활용하여 훈련 진행 및 각 feature들의 중요도 계산
2. Matplotlib 라이브러리를 사용하여 중요도를 막대그래프로 시각화 진행 
<br/>
<br/>

![중요도 시각화](https://user-images.githubusercontent.com/43724177/235898286-629130bf-1835-46a0-864e-a2aca00d8216.PNG)
 
 <br/>
 
### 최적의 feature 갯수 계산
1. scikit-learn 라이브러리의 RFECV를 사용하여 분류 정확도가 가장 높았던 feature의 갯수 및 목록 표시
2. Random forest 와 SVC 를 사용하여 분류 훈련 진행
3. feature 갯수에 따른 분류 정확도를 matplotlib를 사용하여 시각화 
<br/>
<br/>

![feature 갯수-정확도](https://user-images.githubusercontent.com/43724177/235899443-efe11cce-87ae-461b-b41a-a15714e5c405.PNG)

<br/>

### 냉장고의 상황을 잘 반영하는 센서의 위치 유추
1. scikit-learn 라이브러리의 Random forest 와 SVC 를 사용하여 직접 2개의 센서 feature를 바꿔가며 훈련 및 테스트 


### 결론 
1. feature 중요도 
- 시각화에서도 보이듯이 모든 feature가 높은 상관관계를 갖는 것은 아니었음 
- RF 와 SVC 공통적으로 2종류의 feature들이 유독 높은 상관관계를 갖는 것으로 파악됨  
2. 최적의 feature 갯수 계산 
- 적상 유무 분류를 위한 최적의 feature 갯수는 2, 6개로 나타났음
- 적상 유무 + 냉매 부족 분류를 위한 최적의 feature 갯수는 9, 21개로 나타났음
- 1번 과제와 2번 과제에 공통적으로 중요도가 높은 feature가 존재하지만, 과제 2번이 상대적으로 더 어려운 분류과제임을 알 수 있음
3. 최적의 센서 위치 유추
- 두 과제 모두 유리한 센서의 위치가 달랐음

==> 높은 중요도를 갖는 센서 feature를 특정해 낼 수 있었으며, 과제에 따라 다르지만 굳이 44개의 센서 feature를 다 사용하지 않아도 됨
<br/>==> 센서의 최적 위치는 과제에 따라 다른 경향을 보이므로, 두 위치의 절충안을 추가적으로 분석해야할 필요가 있음
