# methane_ANN_modeling

## 1. Analysis preperation

분석 진행 이전에 분석에 필요한 파이썬 패키지를 불러오는 과정이다. 해당 분석에서 활용된 패키지는 아래와 같다.

패키지 목록: pandas, numpy, seaborn, mattplolib, sklearn, statsmodels, tensorflow, keras

<pre><code>#package import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import warnings
warnings.filterwarnings("ignore")
#한글 글꼴
plt.rc("font", family = "Malgun Gothic")
sns.set(font="Malgun Gothic", 
rc={"axes.unicode_minus":False}, style='white')
</pre></code>

## 2. Data importing

분석에 사용되는 데이터를 불러오고 데이터에서 불필요한 컬럼을 제거하였다. 데이터는 건국대에서 측정한 그린피드 데이터를 사용하였다.

<pre><code>df = pd.read_excel("C:/Users/Biolab302/Desktop/SynologyDrive/2024/건국대 메탄과제/GF 데이터 정리_2023 v2.2.xlsx")
print(df.head())
DataFrame = df.drop(columns = ['Period','샘플링일차(1-4)','시간(0-24)','실험구','개체번호'], axis = 1)
DataFrame.head()
</pre></code>

 <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CO2(g/d)</th>
      <th>CH4(g/d)</th>
      <th>H2(g/d)</th>
      <th>CH4(L/d)</th>
      <th>CH4/CO2 ratio</th>
      <th>체중</th>
      <th>DMI(kg/d)</th>
      <th>ECM</th>
      <th>Milk yield(kg/d)</th>
      <th>Milk Fat (%)</th>
      <th>...</th>
      <th>HGB (g/dL)</th>
      <th>HCT (%)</th>
      <th>MCV (fL)</th>
      <th>MCH (pg)</th>
      <th>MCHC (g/dL)</th>
      <th>RDW (%)</th>
      <th>PLT (103/uL)</th>
      <th>MPV (fL)</th>
      <th>PDW</th>
      <th>PCT (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>14348.481103</td>
      <td>398.378871</td>
      <td>0.000000</td>
      <td>606.360535</td>
      <td>0.027765</td>
      <td>573.916667</td>
      <td>24.509917</td>
      <td>23.898418</td>
      <td>31.3</td>
      <td>2.10</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12757.069321</td>
      <td>277.306203</td>
      <td>0.323447</td>
      <td>422.079456</td>
      <td>0.021737</td>
      <td>573.944444</td>
      <td>26.205854</td>
      <td>24.710488</td>
      <td>32.7</td>
      <td>2.04</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14037.750179</td>
      <td>404.333688</td>
      <td>0.797565</td>
      <td>615.424183</td>
      <td>0.028803</td>
      <td>573.944444</td>
      <td>26.205854</td>
      <td>24.710488</td>
      <td>32.7</td>
      <td>2.04</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13734.144812</td>
      <td>331.075822</td>
      <td>0.330569</td>
      <td>503.920581</td>
      <td>0.024106</td>
      <td>573.944444</td>
      <td>26.205854</td>
      <td>24.710488</td>
      <td>32.7</td>
      <td>2.04</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12446.165525</td>
      <td>332.698339</td>
      <td>0.493917</td>
      <td>506.390166</td>
      <td>0.026731</td>
      <td>573.972222</td>
      <td>23.285876</td>
      <td>25.737482</td>
      <td>32.6</td>
      <td>2.25</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 52 columns</p>
</div>

---

이후 분석에 필요한 데이터만 분리하였다.

<pre><code>DataFrame_col = DataFrame.iloc[:,3:8]
DataFrame_col.head()
</pre></code>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CH4(L/d)</th>
      <th>CH4/CO2 ratio</th>
      <th>체중</th>
      <th>DMI(kg/d)</th>
      <th>ECM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>606.360535</td>
      <td>0.027765</td>
      <td>573.916667</td>
      <td>24.509917</td>
      <td>23.898418</td>
    </tr>
    <tr>
      <th>1</th>
      <td>422.079456</td>
      <td>0.021737</td>
      <td>573.944444</td>
      <td>26.205854</td>
      <td>24.710488</td>
    </tr>
    <tr>
      <th>2</th>
      <td>615.424183</td>
      <td>0.028803</td>
      <td>573.944444</td>
      <td>26.205854</td>
      <td>24.710488</td>
    </tr>
    <tr>
      <th>3</th>
      <td>503.920581</td>
      <td>0.024106</td>
      <td>573.944444</td>
      <td>26.205854</td>
      <td>24.710488</td>
    </tr>
    <tr>
      <th>4</th>
      <td>506.390166</td>
      <td>0.026731</td>
      <td>573.972222</td>
      <td>23.285876</td>
      <td>25.737482</td>
    </tr>
  </tbody>
</table>
</div>

---

불러온 데이터에 대한 기술통계량을 확인하였다.

<pre><code>#기술통계량 확인(df)
des_df = round(DataFrame_col.describe(),4)
des_df
</code></pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CH4(L/d)</th>
      <th>CH4/CO2 ratio</th>
      <th>체중</th>
      <th>DMI(kg/d)</th>
      <th>ECM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>352.0000</td>
      <td>346.0000</td>
      <td>352.0000</td>
      <td>352.0000</td>
      <td>352.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>522.8068</td>
      <td>0.0230</td>
      <td>633.4955</td>
      <td>25.7317</td>
      <td>22.9810</td>
    </tr>
    <tr>
      <th>std</th>
      <td>144.0150</td>
      <td>0.0051</td>
      <td>49.6597</td>
      <td>2.4017</td>
      <td>5.0455</td>
    </tr>
    <tr>
      <th>min</th>
      <td>162.0273</td>
      <td>0.0097</td>
      <td>557.0000</td>
      <td>17.6016</td>
      <td>6.6105</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>428.5143</td>
      <td>0.0198</td>
      <td>600.0000</td>
      <td>24.4181</td>
      <td>21.4862</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>496.9553</td>
      <td>0.0226</td>
      <td>628.0000</td>
      <td>26.1367</td>
      <td>23.5542</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>612.0547</td>
      <td>0.0260</td>
      <td>656.3333</td>
      <td>27.3234</td>
      <td>25.6190</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1101.0655</td>
      <td>0.0403</td>
      <td>754.0000</td>
      <td>30.5912</td>
      <td>38.0390</td>
    </tr>
  </tbody>
</table>
</div>

## 3. 이상치 확인

각 데이터에 대한 이상치를 확인하였다. 이상치 확인에는 IQR을 활용한 방법과 Z-score를 활용한 방법을 사용하였다. IQR에 대한 배수는 1.5로 적용하였으며, Z-score에 대한 threshold는 3으로 적용하였다. 실질적으로 이상치를 제거하진 않았으며 분석결과를 토대로 이상치 제거 진행여부를 판단할것이다.

각 독립변수 이상치에 대한 인덱스를 확인하였다.

<pre><code>#이상치 제거
DataFrame_1 = DataFrame_col[['CH4/CO2 ratio']].dropna(axis = 0)
print(DataFrame_1)
def remove_outliers_iqr(data, column_name, threshold=1.5):
    Q1 = np.percentile(data[column_name], 25)  # 1사분위수
    Q3 = np.percentile(data[column_name], 75)  # 3사분위수
    IQR = Q3 - Q1  # IQR 계산
    
    # 이상치를 탐지하여 제거하는 과정
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outlier_indices = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)].index
    data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    
    return data, outlier_indices

Data, Outlier_indices  = remove_outliers_iqr(DataFrame_1, 'CH4/CO2 ratio')
print(Data.describe())
print(Outlier_indices)
</code></pre>

![이상치제거_CH4CO2](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/64374b83-de9d-483d-88b0-1e60ade5d22f)

---

<pre><code>#이상치 제거
#이상치 제거
DataFrame_1 = DataFrame_col[['체중']].dropna(axis = 0)
print(DataFrame_1)
def remove_outliers_iqr(data, column_name, threshold=1.5):
    Q1 = np.percentile(data[column_name], 25)  # 1사분위수
    Q3 = np.percentile(data[column_name], 75)  # 3사분위수
    IQR = Q3 - Q1  # IQR 계산
    
    # 이상치를 탐지하여 제거하는 과정
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outlier_indices = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)].index
    data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    
    return data, outlier_indices

Data, Outlier_indices  = remove_outliers_iqr(DataFrame_1, '체중')
print(Data.describe())
print(Outlier_indices)
</code></pre>

![이상치제거_체중](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/f550c329-3c4a-4480-9ef6-3352a4ed32f8)


---

<pre><code>#이상치 제거
DataFrame_1 = DataFrame_col[['DMI(kg/d)']].dropna(axis = 0)
print(DataFrame_1)
def remove_outliers_iqr(data, column_name, threshold=1.5):
    Q1 = np.percentile(data[column_name], 25)  # 1사분위수
    Q3 = np.percentile(data[column_name], 75)  # 3사분위수
    IQR = Q3 - Q1  # IQR 계산
    
    # 이상치를 탐지하여 제거하는 과정
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outlier_indices = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)].index
    data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    
    return data, outlier_indices

Data, Outlier_indices  = remove_outliers_iqr(DataFrame_1, 'DMI(kg/d)')
print(Data.describe())
print(Outlier_indices)
</code></pre>

![이상치제거_DMI](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/dd08c3ff-5538-4b3e-9cdb-5446e5225d79)


---

<pre><code>#이상치 제거
DataFrame_1 = DataFrame_col[['ECM']].dropna(axis = 0)
print(DataFrame_1)
def remove_outliers_iqr(data, column_name, threshold=1.5):
    Q1 = np.percentile(data[column_name], 25)  # 1사분위수
    Q3 = np.percentile(data[column_name], 75)  # 3사분위수
    IQR = Q3 - Q1  # IQR 계산
    
    # 이상치를 탐지하여 제거하는 과정
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outlier_indices = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)].index
    data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    
    return data, outlier_indices

Data, Outlier_indices  = remove_outliers_iqr(DataFrame_1, 'ECM')
print(Data.describe())
print(Outlier_indices)
</code></pre>

![이상치제거_ECM](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/a140b612-66f7-4939-b887-1beb63c8c8fa)


---

이후 종속변수에 대한 이상치도 확인하였다.

<pre><code># DataFrame에서 종속 변수 열 선택
dependent_variable = DataFrame['CH4(L/d)']

# 이상치 제거 기준 설정 (예: Z-score를 사용한 방법)
z_scores = np.abs((dependent_variable - dependent_variable.mean()) / dependent_variable.std())
threshold = 3  # 임계값 설정
print(z_scores)
# 이상치의 인덱스 확인
outlier_indices = z_scores[z_scores > threshold].index
print("이상치의 인덱스:")
print(outlier_indices)

</code></pre>

![이상치제거_CHr](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/a1c6ad87-95c4-4fe6-90da-8a4c8fd3e540)


## 4. 데이터 정규화(Data normalization)

데이터 간의 스케일이 다른경우 계산의 용이함을 위하여 데이터 정규화를 진행한다. 데이터 정규화 시에 데이터 처리속도가 계선된다. 데이터 정규화에는 Min-Max normalization 기법을 사용하였다.

<pre><code>#Data nomalization
nor_df = (DataFrame_col - DataFrame_col.min()) / (DataFrame_col.max() - DataFrame_col.min())
# Undo normalization for y value
y_unnormalized = nor_df['CH4(L/d)'] * (DataFrame_col['CH4(L/d)'].max() - DataFrame_col['CH4(L/d)'].min()) + DataFrame_col['CH4(L/d)'].min()
# Remove flow column from nor_df
nor_df.drop('CH4(L/d)', axis=1, inplace=True)
# Add y_unnormalized column to nor_df
nor_df['CH4(L/d)'] = y_unnormalized
print(nor_df.describe())
</code></pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CH4/CO2 ratio</th>
      <th>체중</th>
      <th>DMI(kg/d)</th>
      <th>ECM</th>
      <th>CH4(L/d)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>346.000000</td>
      <td>352.000000</td>
      <td>352.000000</td>
      <td>352.000000</td>
      <td>352.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.435934</td>
      <td>0.388302</td>
      <td>0.625894</td>
      <td>0.520881</td>
      <td>522.806765</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.166701</td>
      <td>0.252080</td>
      <td>0.184897</td>
      <td>0.160538</td>
      <td>144.015043</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>162.027258</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.329349</td>
      <td>0.218274</td>
      <td>0.524766</td>
      <td>0.473318</td>
      <td>428.514293</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.421301</td>
      <td>0.360406</td>
      <td>0.657075</td>
      <td>0.539120</td>
      <td>496.955311</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.535041</td>
      <td>0.504230</td>
      <td>0.748435</td>
      <td>0.604817</td>
      <td>612.054699</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1101.065482</td>
    </tr>
  </tbody>
</table>
</div>

## 5. Feature selection

Feature selection은 Pearson's correlation matrix와 vif score 분석을 사용한다. 하지만 우리는 기존 일본 추정식 (2)의 변수인 CH4/CO2 ratio, 체중, ECM, CH4(L/d)에 대한 결과만 중점적으로 확인하였다.

기존 일본 추정식 (2): CH4(L/day)= −507(45.0) + 0.536(0.0635)LW + 8.76(0.630)ECM + 5,029(313.8)CH4/CO2

### 5-1. Pearson's correlation matrix

Pearson's correlation matrix는 변수간의 상관관계를 확인하여 선택하는 통계적 방법이다. 각 변수간의 상관계수를 통해 상관관계를 확인하였다.

<pre><code>#변수간 Correlation 확인
plt.figure(figsize = (14, 10))
sns.heatmap(nor_df.corr(), annot = True)
</code></pre>

![correlation matrix](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/9bba3b68-aace-4314-9e7a-a7b037929268)


### 5-2. VIF score analysis

VIF는 독립변수간 상관 관계를 측정하는 척도이다. VIF score analysis는 독립변수 간 VIF를 확인하여 다중공선성을 파악하는 분석방법이다. VIF가 10이 넘으면 다중공선성이 있다고 판단하며 5가 넘으면 주의할 필요가 있는것으로 본다.

![VIF 계산식](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/744ce0b5-738e-40d5-a359-44a34e94e6b5)


아래 코드는 기존 일본 추정식에서 사용된 변수 간의 VIF를 비교한 코드이다.

<pre><code>#변수 구분
x = nor_df.drop(columns = ['CH4(L/d)','DMI(kg/d)'], axis =1)
x = x.dropna(axis=0).reset_index(drop=True)
y = nor_df[['CH4(L/d)']]
# 독립변수 행렬 생성
matrix_x = np.array(x)
#VIF 점수 계산
vif = pd.DataFrame()
vif['Features'] = x.columns
vif['VIF Score'] = [variance_inflation_factor(matrix_x,i) for i in range(matrix_x.shape[1])]

vif
</code></pre>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Features</th>
      <th>VIF Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CH4/CO2 ratio</td>
      <td>4.446800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>체중</td>
      <td>3.733935</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ECM</td>
      <td>6.721781</td>
    </tr>
  </tbody>
</table>
</div>

---

## 6. Model selection

선택된 변수를 활용하여 변수모델을 선택하는 과정이다. 일본 추정식 (2)을 기반으로 모델식을 구축하고 있으므로 일본 추정식 (2)에 맞춰 모델을 선택하였다.

선택된 변수모델: CH4(L/d) = CH4/CO2 + 체중 + ECM

## 7. Model Fitting & Model evaluation

선택된 변수 모델에 데이터를 학습시키는 과정이다. 해당 데이터를 MLR, ML, ANN을 사용하여 모델을 구축하였다.

---

### 7-1 결측치 제거

변수모델에 맞춰 결측치 인덱스를 제거하였다. 아래 결과는 결측치 제거한 후 변수의 기술통계량을 나타내었다.

<pre><code>#결측치 제거
df_drop = nor_df.drop(columns = ['DMI(kg/d)'], axis = 1)
df_na = df_drop.dropna(axis=0).reset_index(drop=True)
x = df_na.drop(columns = ['CH4(L/d)'], axis = 1)
y = df_na[['CH4(L/d)']]
x.describe()
</code></pre>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CH4/CO2 ratio</th>
      <th>체중</th>
      <th>ECM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>346.000000</td>
      <td>346.000000</td>
      <td>346.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.435934</td>
      <td>0.388885</td>
      <td>0.520441</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.166701</td>
      <td>0.252987</td>
      <td>0.161418</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.329349</td>
      <td>0.225888</td>
      <td>0.473318</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.421301</td>
      <td>0.360406</td>
      <td>0.539120</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.535041</td>
      <td>0.507614</td>
      <td>0.604051</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

---

### 7-2 Data split

ML, ANN 분석을 위해서, training data와 test data로 split하는 과정이 필수적이다. Data split은 8:2의 비율로 진행하였으며, randome_state는 42로 지정하였다.

<pre><code>#Data split
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x, y, test_size = 0.2, random_state = 42)

# training data set의 인덱스 초기화
x_train_1 = x_train_1.reset_index(drop=True)
y_train_1 = y_train_1.reset_index(drop=True)
</code></pre>

---

### 7-3 MLR 분석

전체 데이터를 활용하여 MLR 분석을 진행하였다. MLR 분석에는 scikit-learn의 LinearRegression을 활용하여 진행하였다. 모델평가지표로는 Accuracy, RMSE, RRMSE, R2, adjusted R2를 사용하였다.

<pre><code>#MLR(machine learning x)
def MLR(x, y):
    model = LinearRegression()
    model.fit(x, y)
    
    # Get coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_
    
    MAPE = np.mean(100 * (np.abs(y-model.predict(x))/y))
    accuracy = 100 - MAPE
    # Calculating RMSE
    y_pred = model.predict(x)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    # Calculating Relative RMSE
    relative_rmse = (rmse / np.mean(y))*100
    # Calculating R-squared
    r2 = r2_score(y, y_pred)
    # Calculating adj-R2
    adj_r2 = 1 - (1-r2)*(len(x)-1)/(len(x)-x.shape[1]-1)
    # Printing the results of the current fold iteration
    print('Coefficient:', coefficients)
    print('Intercept:', intercept)
    print('Accuracy:', accuracy, 'RMSE', rmse, 'RRMSE', relative_rmse, 'r2', r2, 'adj_r2', adj_r2)

MLR_1 = MLR(x,y)

</code></pre>

![MLR](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/9dce7d3b-daa8-48de-8841-cdc6865b54ca)


#### MLR 결과

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: middle">
      <th>모델</th>
      <th>모델식</th>
      <th>Accuracy</th>
      <th>R2</th>
      <th>Adjr2</th>
      <th>RMSE</th>
      <th>RRMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Model1</th>
      <td>CH4(L/d) = 101.68 + 777.40 CH4/CO2 + 90.54 체중 + 88.17 ECM</td>
      <td>90.63</td>
      <td>0.81</td>
      <td>0.81</td>
      <td>62.53</td>
      <td>11.99</td>
    </tr>
  </tbody>
</table>
</div>

---

### 7-4 ML 분석

ML 분석에는 8:2의 비율로 스플릿된 데이터를 사용하여 진행하였다. 10-fold crossvalidation을 진행하여 모델을 비교검증하였다. 모델평가는 Accuracy, RMSE, RRMSE, R2, adjusted R2로 평가하였다.

<pre><code>
def Multiple(x_train, y_train, x_test, y_test, k_fold=10):
    SearchResultsData=pd.DataFrame()
    # Create MLR model
    model = LinearRegression()
     # Perform k-fold cross validation
    kf = KFold(n_splits=k_fold, shuffle=True, random_state = 42)
    fold_number = 1
    for train_index, val_index in kf.split(x_train):
        X_train_fold, X_val_fold = x_train.loc[train_index], x_train.loc[val_index]
        Y_train_fold, Y_val_fold = y_train.loc[train_index], y_train.loc[val_index]
        
        # Fitting MLR to the Training set
        model.fit(X_train_fold, Y_train_fold)
        
        # Get coefficients and intercept
        coefficients = model.coef_
        intercept = model.intercept_
        
        MAPE_val = np.mean(100 * (np.abs(Y_val_fold-model.predict(X_val_fold))/Y_val_fold))
        MAPE = np.mean(100 * (np.abs(y_test-model.predict(x_test))/y_test))
        accuracy_val = 100 - MAPE_val
        accuracy = 100 - MAPE
        # Calculating RMSE
        y_pred_val = model.predict(X_val_fold)
        y_pred = model.predict(x_test)
        rmse_val = np.sqrt(np.mean((Y_val_fold - y_pred_val)**2))
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        # Calculating Relative RMSE
        relative_rmse_val = (rmse_val / np.mean(Y_val_fold))*100
        relative_rmse = (rmse / np.mean(y_test))*100
        # Calculating R-squared
        r2_val = r2_score(Y_val_fold, y_pred_val)
        r2_test = r2_score(y_test, y_pred)
        # Calculating adj-R2
        adj_r2_val = 1 - (1-r2_val)*(len(X_val_fold)-1)/(len(X_val_fold)-X_val_fold.shape[1]-1)
        adj_r2_test = 1 - (1-r2_test)*(len(x_test)-1)/(len(x_test)-x_test.shape[1]-1)
        # Printing the results of the current fold iteration
        print('Fold:', fold_number)
        print('Coefficient:', coefficients)
        print('Intercept:', intercept)
        print('Accuracy_val:', accuracy_val,'accuracy_test',accuracy, 'RMSE_val:', rmse_val, 'RRMSE_val',relative_rmse_val, 
              'RMSE_test',rmse,'RRMSE_test',relative_rmse,'R2_val',r2_val,'r2_test',r2_test,'adjr2_val',adj_r2_val,'adjr2_test',adj_r2_test)
        fold_number += 1
        # Appending the results to the dataframe
        SearchResultsData = pd.concat([SearchResultsData,pd.DataFrame(data=[[fold_number,coefficients, intercept, 
        accuracy_val, accuracy, rmse_val,relative_rmse_val, rmse, relative_rmse, r2_val, r2_test,adj_r2_val,adj_r2_test]],
        columns=['Fold', 'Coefficients', 'intercept', 'Accuracy_val', 'Accuracy_test', 'RMSE_val','RRMSE_val', 'RMSE_test', 
        'RRMSE_test','r2_val','r2_test','adjr2_val','adjr2_test'])])
    
    return(SearchResultsData)

#Calling the function
ML_1 = Multiple(x_train_1, y_train_1, x_test_1, y_test_1, k_fold=10)
ML_1.to_excel('CH4_ML1.xlsx', index=False)

</code></pre>

![ML 결과 정리](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/7caa174a-384d-4cd6-aa48-258718c3c36d)


---

#### 7-4-1 ML 결과 (validation data)

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: middle;">
      <th>Model</th>
      <th>Accuracy</th>
      <th>R2</th>
      <th>Adjr2</th>
      <th>RMSE</th>
      <th>RRMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중 + ECM</th>
      <td>90.95</td>
      <td>0.80</td>
      <td>0.77</td>
      <td>63.34</td>
      <td>12.84</td>
    </tr>
  </tbody>
  </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>91.08</td>
      <td>0.84</td>
      <td>0.82</td>
      <td>67.23</td>
      <td>11.68</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>91.34</td>
      <td>0.86</td>
      <td>0.84</td>
      <td>59.04</td>
      <td>10.74</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>92.33</td>
      <td>0.74</td>
      <td>0.71</td>
      <td>50.43</td>
      <td>9.51</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>89.42</td>
      <td>0.74</td>
      <td>0.71</td>
      <td>71.15</td>
      <td>14.55</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>92.46</td>
      <td>0.83</td>
      <td>0.81</td>
      <td>55.09</td>
      <td>10.84</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>90.99</td>
      <td>0.77</td>
      <td>0.74</td>
      <td>66.25</td>
      <td>12.59</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>85.75</td>
      <td>0.84</td>
      <td>0.82</td>
      <td>77.08</td>
      <td>14.74</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>88.00</td>
      <td>0.53</td>
      <td>0.47</td>
      <td>81.10</td>
      <td>16.07</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>89.82</td>
      <td>0.67</td>
      <td>0.62</td>
      <td>61.54</td>
      <td>12.55</td>
    </tr>
  </tbody>
</table>
</div>

---

#### 7-4-2 ML 결과 (test data)

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: middle;">
      <th>Model</th>
      <th>Accuracy</th>
      <th>R2</th>
      <th>Adjr2</th>
      <th>RMSE</th>
      <th>RRMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중 + ECM</th>
      <td>91.28</td>
      <td>0.84</td>
      <td>0.84</td>
      <td>56.60</td>
      <td>10.64</td>
    </tr>
  </tbody>
  </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>91.45</td>
      <td>0.84</td>
      <td>0.84</td>
      <td>56.48</td>
      <td>10.62</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>91.47</td>
      <td>0.84</td>
      <td>0.84</td>
      <td>56.51</td>
      <td>10.63</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>91.22</td>
      <td>0.84</td>
      <td>0.84</td>
      <td>56.71</td>
      <td>10.66</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>91.40</td>
      <td>0.85</td>
      <td>0.84</td>
      <td>56.04</td>
      <td>10.54</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>91.40</td>
      <td>0.85</td>
      <td>0.84</td>
      <td>56.14</td>
      <td>10.56</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>91.28</td>
      <td>0.84</td>
      <td>0.84</td>
      <td>56.33</td>
      <td>10.59</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>91.45</td>
      <td>0.84</td>
      <td>0.83</td>
      <td>56.78</td>
      <td>10.68</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>91.05</td>
      <td>0.84</td>
      <td>0.83</td>
      <td>57.05</td>
      <td>10.73</td>
    </tr>
  </tbody>
    </thead>
  <tbody>
    <tr>
      <th>CH4/CO2 + 체중+ ECM</th>
      <td>91.37</td>
      <td>0.85</td>
      <td>0.84</td>
      <td>55.88</td>
      <td>10.51</td>
    </tr>
  </tbody>
</table>
</div>

---

#### 7-4-3 ML 모델 제시

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: middle">
      <th>모델</th>
      <th>모델식</th>
      <th>Accuracy</th>
      <th>R2</th>
      <th>Adjr2</th>
      <th>RMSE</th>
      <th>RRMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Model1</th>
      <td>CH4(L/d) = 97.22 + 776.37 CH4/CO2 + 95.40 체중 + 93.62 ECM</td>
      <td>91.40</td>
      <td>0.85</td>
      <td>0.84</td>
      <td>56.14</td>
      <td>10.56</td>
    </tr>
  </tbody>
</table>
</div>

ML 결과 중 모델 평가지표를 비교하여 모델을 제시하였다. valdation data에 대한 Accuracy가 92.46으로 가장 높았으며, RMSE도 55.09로 두번째로 낮았다. test data에 대한 결과를 비교하였을때 validation data의 결과와 큰 차이를 보이지 않아 해당 모델을 제시하였다.

---

### 7-5 ANN 분석

#### 7-5-1 Kerastuner

Kerastuner를 사용하여 ANN 모델의 최적의 Dense layer 수와 각 Dense layer별 nod 수를 확인하였다. Dense layer 수는 1-5로 설정하였다. 각 Dense layer별 노드의 범위는 32-512로 설정하였으며, 32씩 증가하도록 설정하였다.

<pre><code>import kerastuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), 
                    activation='relu', input_shape=(x_train_1.shape[1],)))
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                        activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=5,
    directory='CH4_modeling_1',
    project_name='CH4_model_1')

tuner.search(x_train_1, y_train_1, epochs=10, validation_data=(x_test_1, y_test_1))

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal number of layers is {best_hps.get('num_layers')}.
""")

#layer 별 노드개수(model1)
print("layer 1:", best_hps.get('units_0'), "layer 2", best_hps.get("units_1"), "layer 3", best_hps.get("units_2"),
      "layer 4", best_hps.get("units_3"), "layer 5", best_hps.get("units_4"))
</code>
</pre>

| Dense Layer 수 |                                     Layer별 노드 수                                      |
| :------------: | :--------------------------------------------------------------------------------------: |
|       5        | input – Dense1(352) - Dense2(288) - Dense3(480) <br>- Dense4(512) - Dense5(192) - output |

kerastuner 결과 최적의 Dense Layer 수는 5개로 나타났으며, Layer별 노드수는 각각 352개, 288개, 480개, 512개, 192개 (Dense1-Dense5)로 나타났다. 해당 결과를 적용하여 hyperparameter tuning을 진행하였다.

---

#### 7-5-2 hyperparameter tuning(batch_size, epoch)

Kerastuner 결과를 적용하여 batch_size와 epoch을 tuning하였다. batch_size와 epoch에 대한 list는 아래 표에 정리하였다. 해당 리스트에서 최적의 batch_size와 epoch을 확인하고자 하였다.

|        batch_size         |          epoch           |
| :-----------------------: | :----------------------: |
| 5, 10, 15, 20, 25, 30, 35 | 5, 10, 50, 100, 250, 500 |

<pre><code>
# Defining a function to find the best parameters for ANN and obtain results for the training dataset
def FunctionFindBestParams_1(x_train, y_train, x_test, y_test, k_fold=10):
    
    # Defining the list of hyper parameters to try
    batch_size_list=[5, 10, 15, 20, 25, 30, 35]
    epoch_list  =   [5, 10, 50, 100, 250, 500]
    
    SearchResultsData=pd.DataFrame()
    
    # Initializing the trials
    TrialNumber=0
    for batch_size_trial in batch_size_list:
        for epochs_trial in epoch_list:
            TrialNumber+=1
            model = tuner.hypermodel.build(best_hps)
            # Perform k-fold cross validation
            kf = KFold(n_splits=k_fold, shuffle=True, random_state = 42)
            fold_number = 1
            for train_index, val_index in kf.split(x_train):
                X_train_fold, X_val_fold = x_train.loc[train_index], x_train.loc[val_index]
                Y_train_fold, Y_val_fold = y_train.loc[train_index], y_train.loc[val_index]
                
                # Fitting the ANN to the Training set
                model.fit(X_train_fold, Y_train_fold ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)
                MAPE_val = np.mean(100 * (np.abs(Y_val_fold-model.predict(X_val_fold))/Y_val_fold))
                MAPE = np.mean(100 * (np.abs(y_test-model.predict(x_test))/y_test))
                accuracy_val = 100 - MAPE_val
                accuracy = 100 - MAPE
                # Calculating RMSE
                y_pred_val = model.predict(X_val_fold)
                y_pred = model.predict(x_test)
                rmse_val = np.sqrt(np.mean((Y_val_fold - y_pred_val)**2))
                rmse = np.sqrt(np.mean((y_test - y_pred)**2))
                # Calculating Relative RMSE
                relative_rmse_val = (rmse_val / np.mean(Y_val_fold))*100
                relative_rmse = (rmse / np.mean(y_test))*100
                # Calculating R-squared
                r2_val = r2_score(Y_val_fold, y_pred_val)
                r2_test = r2_score(y_test, y_pred)
                # Calculating adj-R2
                adj_r2_val = 1 - (1-r2_val)*(len(X_val_fold)-1)/(len(X_val_fold)-X_val_fold.shape[1]-1)
                adj_r2_test = 1 - (1-r2_test)*(len(x_test)-1)/(len(x_test)-x_test.shape[1]-1)
                # Printing the results of the current fold iteration
                print('Fold:', fold_number, 'TrialNumber:', TrialNumber, 'Parameters:', 'batch_size:',
                      batch_size_trial, '-', 'epochs:', epochs_trial, 'Accuracy_val:', accuracy_val,
                      'accuracy_test',accuracy, 'RMSE_val:', rmse_val, 'RRMSE_val', relative_rmse_val,
                      'RMSE_test', rmse, 'RRMSE_test', relative_rmse, 'R2_val', r2_val, 'r2_test', r2_test,
                     'adjr2_val',adj_r2_val,'adjr2_test',adj_r2_test)
                
                fold_number += 1
            
            
                # Appending the results to the dataframe
                SearchResultsData = pd.concat([SearchResultsData,
                                               pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial),
                                                                   accuracy_val, accuracy, rmse_val, relative_rmse_val, rmse,
                                                                   relative_rmse, r2_val, r2_test,adj_r2_val,adj_r2_test]],
                                                            columns=['TrialNumber', 'Parameters', 'Accuracy_val', 'Accuracy_test',
                                                                     'RMSE_val', 'RRMSE_val', 'RMSE_test', 'RRMSE_test',
                                                                    'r2_val','r2_test','adjr2_val','adjr2_test'])])
                
    return(SearchResultsData)

# Calling the function
ANN_1 = FunctionFindBestParams_1(x_train_1, y_train_1, x_test_1, y_test_1, k_fold=10)
ANN_1.to_excel('CH4_ANN_model1.xlsx', index=False)
</code></pre>

![hyperparameter tuning](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/5ae13d10-45df-4684-ba31-4fe69415f4d4)


---

#### 7-5-3 ANN 결과(validation data)

|         모델         | Hyperparameters <br> (batch size – epoch) | Accuracy |  R2  | Adjr2 | RMSE  | RRMSE |
| :------------------: | :---------------------------------------: | :------: | :--: | :---: | :---: | :---: |
| CH4/CO2 + 체중 + ECM |                  30-250                   |  93.28   | 0.92 | 0.91  | 43.53 | 7.92  |
| CH4/CO2 + 체중 + ECM |                   15-50                   |  91.90   | 0.89 | 0.88  | 51.82 | 9.42  |
| CH4/CO2 + 체중 + ECM |                  10-250                   |  91.89   | 0.89 | 0.87  | 52.44 | 9.54  |
| CH4/CO2 + 체중 + ECM |                   35-50                   |  91.79   | 0.88 | 0.87  | 54.08 | 9.84  |
| CH4/CO2 + 체중 + ECM |                   35-50                   |  92.57   | 0.84 | 0.82  | 52.92 | 10.42 |
| CH4/CO2 + 체중 + ECM |                  20-100                   |  92.18   | 0.84 | 0.82  | 52.94 | 10.42 |
| CH4/CO2 + 체중 + ECM |                   10-5                    |  92.37   | 0.84 | 0.82  | 53.40 | 10.51 |
| CH4/CO2 + 체중 + ECM |                   15-5                    |  92.23   | 0.84 | 0.82  | 53.40 | 10.51 |
| CH4/CO2 + 체중 + ECM |                   30-10                   |  92.47   | 0.84 | 0.82  | 53.48 | 10.52 |
| CH4/CO2 + 체중 + ECM |                   20-5                    |  92.52   | 0.84 | 0.82  | 53.69 | 10.57 |
| CH4/CO2 + 체중 + ECM |                   5-100                   |  92.64   | 0.84 | 0.82  | 54.03 | 10.63 |
| CH4/CO2 + 체중 + ECM |                   5-50                    |  92.42   | 0.84 | 0.82  | 54.04 | 10.64 |

---

#### 7-5-4 ANN 결과(test data)

|         모델         | Hyperparameters <br> (batch size – epoch) | Accuracy |  R2  | Adjr2 | RMSE  | RRMSE |
| :------------------: | :---------------------------------------: | :------: | :--: | :---: | :---: | :---: |
| CH4/CO2 + 체중 + ECM |                  30-250                   |  91.39   | 0.85 | 0.84  | 55.49 | 10.44 |
| CH4/CO2 + 체중 + ECM |                   15-50                   |  91.22   | 0.84 | 0.83  | 56.80 | 10.68 |
| CH4/CO2 + 체중 + ECM |                  10-250                   |  91.66   | 0.86 | 0.86  | 52.86 | 9.94  |
| CH4/CO2 + 체중 + ECM |                   35-50                   |  91.26   | 0.84 | 0.84  | 56.64 | 10.65 |
| CH4/CO2 + 체중 + ECM |                   35-50                   |  91.64   | 0.84 | 0.84  | 56.73 | 10.67 |
| CH4/CO2 + 체중 + ECM |                  20-100                   |  91.79   | 0.86 | 0.85  | 54.37 | 10.22 |
| CH4/CO2 + 체중 + ECM |                   10-5                    |  91.15   | 0.84 | 0.84  | 56.39 | 10.61 |
| CH4/CO2 + 체중 + ECM |                   15-5                    |  91.08   | 0.84 | 0.83  | 56.87 | 10.70 |
| CH4/CO2 + 체중 + ECM |                   30-10                   |  91.52   | 0.84 | 0.84  | 56.38 | 10.60 |
| CH4/CO2 + 체중 + ECM |                   20-5                    |  91.33   | 0.84 | 0.84  | 56.65 | 10.65 |
| CH4/CO2 + 체중 + ECM |                   5-100                   |  92.06   | 0.85 | 0.84  | 55.24 | 10.39 |
| CH4/CO2 + 체중 + ECM |                   5-50                    |  91.59   | 0.84 | 0.84  | 56.50 | 10.62 |

---

#### 7-5-5 ANN 모델 제시

|  모델   | Hyperparameters | Accuracy |  R2  | Adfr2 | RMSE  |
| :-----: | :-------------: | :------: | :--: | :---: | :---: |
| Model 1 |     30-250      |  93.28   | 0.92 | 0.91  | 43.53 |

Validation data에 대하여 생성된 모델 중 Accuracy가 93.28, R2가 0.92, Adjusted R2가 0.91로 가장 높았으며, RMSE가 43.53으로 가장 낮았다. test data에 대한 결과와 비교해보았을 때도 큰 차이가 없어 해당 모델을 제시하였다.

### 7-6 모델간의 비교

| 모델 | Accuracy |  R2  | Adjr2 | RMSE  | RRMSE |
| :--: | :------: | :--: | :---: | :---: | :---: |
| MLR  |  90.63   | 0.81 | 0.81  | 62.53 | 11.99 |
|  ML  |  91.40   | 0.85 | 0.84  | 56.04 | 10.54 |
| ANN  |  93.28   | 0.92 | 0.91  | 43.53 | 7.92  |

MLR, ML, ANN 모델을 평가지표로 비교하였다. ANN 모델이 MLR 모델보다 2.65, ML 모델보다 1.88 향상된 Accuracy를 보였다. Adjusted R2에서도 ANN 모델이 MLR과 ML보다 각각 0.10, 0.07 향상되었다. 평가지표를 비교하였을때 ANN 모델이 MLR, ML보다 성능이 향상됨을 보였다.

# 8. 일본 추정식 (1)의 변수를 활용한 분석
일본 추정식(1)의 변수(CH4/CO2, 체중, DMI, ECM)을 활용하여 분석을 진행하였다.

## 8-1 기술통계량 확인
문서 내 [2. Data importing](#기술통계량)에서의 기술통계량과 동일하다.

## 8-2 이상치 확인
문서 내 [3. 이상치 확인](#3-이상치-확인)과 동일하게 진행되었다. 변수별 이상치를 확인하였으나 따로 이상치 제거는 진행하지 않았다.

## 8-3 데이터 정규화
문서 내 [4. Nomralization](#4-데이터-정규화)와 동일하게 진행되었다.

## 8-4 Feature selection
변수는 4가지를 선택하여 진행하였기 때문에 변수간 Correlation과 Vif score만 확인하였다.

### 8-4-1 Pearson's correlation matrix
문서 내 [5-1. Pearsons correlation matrix](#5-1-pearsons-correlation-matrix)와 동일하게 진행되었다.

### 8-4-2 VIF score analysis
4가지 독립변수(CH4/CO2, 체중, DMI, ECM)와 관련하여 VIF score anlaysis를 진행하였다. DMI에 대한 VIF score가 10.75로 10보다 크게 나타나 다중공선성을 위배하였다고 판단되었다. 따로 변수를 제거하지 않고 이후 분석을 진행하였다.

|Features|VIF Score|
|--------|---------|
|CH4/CO2 ratio|5.29|
|체중|4.20|
|DMI(kg/d)|10.75|
|ECM|9.54|

## 8-5 결측치 제거
결측치가 존재하는 인덱스를 제거하였다. 결측치를 제거한 후 346개의 데이터가 확인되었다.
<pre><code>#결측치 제거
df_na_1 = nor_df.dropna(axis=0).reset_index(drop=True)
print(df_na_1.describe())
x_1 = df_na_1.drop(columns = ['CH4(L/d)'], axis = 1)
print(x_1.describe())
y_1 = df_na_1[['CH4(L/d)']]
</code></pre>

## 8-6 데이터 split
데이터 split은 8:2로 진행하였으며, random_state는 42로 지정하였다
<pre><code>
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_1, y_1, test_size = 0.2, random_state = 42)
x_train_2 = x_train_2.reset_index(drop=True)
y_train_2 = y_train_2.reset_index(drop=True)
</code></pre>

## 8-7 MLR 분석결과
<pre><code>
#MLR(machine learning x)
def MLR(x, y):
    model = LinearRegression()
    model.fit(x, y)
    
    # Get coefficients and intercept
    coefficients = model.coef_
    intercept = model.intercept_
    
    MAPE = np.mean(100 * (np.abs(y-model.predict(x))/y))
    accuracy = 100 - MAPE
    # Calculating RMSE
    y_pred = model.predict(x)
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    # Calculating Relative RMSE
    relative_rmse = (rmse / np.mean(y))*100
    # Calculating R-squared
    r2 = r2_score(y, y_pred)
    # Calculating adj-R2
    adj_r2 = 1 - (1-r2)*(len(x)-1)/(len(x)-x.shape[1]-1)
    # Printing the results of the current fold iteration
    print('Coefficient:', coefficients)
    print('Intercept:', intercept)
    print('Accuracy:', accuracy, 'RMSE', rmse, 'RRMSE', relative_rmse, 'r2', r2, 'adj_r2', adj_r2)

MLR_2 = MLR(x_1,y_1)
</code></pre>

### MLR 분석결과
MLR 분석결과 Accuracy 90.72, RMSE 62.08, RRMSE 11.90, R2 0.81, AdjR2 0.81가 확인되었다.
|모델|Accuracy|RMSE|RRMSE|R2|Adjr2|
|----|--------|----|-----|--|-----|
|CH4(L/d)= 81.45 +778.214 CH4/CO2 +80.15 체중 +43.78 DMI +81.86 ECM|90.72|62.08|11.90|0.81|0.81|

## 8-8 ML 분석결과
<pre><code>
def Multiple(x_train, y_train, x_test, y_test, k_fold=10):
    SearchResultsData=pd.DataFrame()
    # Create MLR model
    model = LinearRegression()
     # Perform k-fold cross validation
    kf = KFold(n_splits=k_fold, shuffle=True, random_state = 42)
    fold_number = 1
    for train_index, val_index in kf.split(x_train):
        X_train_fold, X_val_fold = x_train.loc[train_index], x_train.loc[val_index]
        Y_train_fold, Y_val_fold = y_train.loc[train_index], y_train.loc[val_index]
        
        # Fitting MLR to the Training set
        model.fit(X_train_fold, Y_train_fold)
        
        # Get coefficients and intercept
        coefficients = model.coef_
        intercept = model.intercept_
        
        MAPE_val = np.mean(100 * (np.abs(Y_val_fold-model.predict(X_val_fold))/Y_val_fold))
        MAPE = np.mean(100 * (np.abs(y_test-model.predict(x_test))/y_test))
        accuracy_val = 100 - MAPE_val
        accuracy = 100 - MAPE
        # Calculating RMSE
        y_pred_val = model.predict(X_val_fold)
        y_pred = model.predict(x_test)
        rmse_val = np.sqrt(np.mean((Y_val_fold - y_pred_val)**2))
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        # Calculating Relative RMSE
        relative_rmse_val = (rmse_val / np.mean(Y_val_fold))*100
        relative_rmse = (rmse / np.mean(y_test))*100
        # Calculating R-squared
        r2_val = r2_score(Y_val_fold, y_pred_val)
        r2_test = r2_score(y_test, y_pred)
        # Calculating adj-R2
        adj_r2_val = 1 - (1-r2_val)*(len(X_val_fold)-1)/(len(X_val_fold)-X_val_fold.shape[1]-1)
        adj_r2_test = 1 - (1-r2_test)*(len(x_test)-1)/(len(x_test)-x_test.shape[1]-1)
        # Printing the results of the current fold iteration
        print('Fold:', fold_number)
        print('Coefficient:', coefficients)
        print('Intercept:', intercept)
        print('Accuracy_val:', accuracy_val,'accuracy_test',accuracy, 'RMSE_val:', rmse_val, 'RRMSE_val',relative_rmse_val, 
              'RMSE_test',rmse,'RRMSE_test',relative_rmse,'R2_val',r2_val,'r2_test',r2_test,'adjr2_val',adj_r2_val,'adjr2_test',adj_r2_test)
        fold_number += 1
        # Appending the results to the dataframe
        SearchResultsData = pd.concat([SearchResultsData,
                                       pd.DataFrame(data=[[fold_number,coefficients, intercept, accuracy_val, accuracy, rmse_val,
                                                           relative_rmse_val, rmse, relative_rmse, r2_val, r2_test,adj_r2_val,adj_r2_test]],
                                                    columns=['Fold', 'Coefficients', 'intercept', 'Accuracy_val', 'Accuracy_test', 'RMSE_val',
                                                             'RRMSE_val', 'RMSE_test', 'RRMSE_test','r2_val','r2_test','adjr2_val','adjr2_test'])])
    
    return(SearchResultsData)

ML_2 = Multiple(x_train_2, y_train_2, x_test_2, y_test_2, k_fold=10)
ML_2.to_excel('CH4_ML2.xlsx', index=False)
</code></pre>

### ML 분석결과(Validation data)
![image](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/e42842bb-17b3-4bc3-8d31-c484dfe69e90)

### ML 분석결과(Test data)
![image](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/02ded89f-3d51-42c1-b45d-b7727a56fa9d)


## ANN 결과정리
![image](https://github.com/inpyo03/methane_ANN_modeling/assets/160727249/859540cd-cd83-43f1-be78-06b90c032871)

