import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import time

# 시작 시간 기록
start_time = time.time()
file_path = '3month.xlsx'  
# 데이터 프레임 불러오기
df_sheet1 = pd.read_excel(file_path, engine='openpyxl', sheet_name='전기사용량_3개월분', index_col=0, skiprows=1, header=[0,1])
df_sheet1 = df_sheet1.dropna(axis=1)

df_sheet2 = pd.read_excel(file_path, engine='openpyxl', sheet_name='수도사용량_3개월분', index_col=0, skiprows=1, header=[0,1])
df_sheet2 = df_sheet2.dropna(axis=0)

df_sheet1 = df_sheet1.iloc[:, :368]

# 여기서부터 데이터 생성 반복
all_final_data = pd.DataFrame()  # 빈 데이터프레임 생성
for i in range(0,5):
    col_name = f'ho_{i}'  # 열 이름 생성
    ho_i = df_sheet1.iloc[i]  # 해당 열 선택
    ho_i = pd.DataFrame(ho_i)  # 데이터프레임으로 변환
    ho_i = ho_i.droplevel(level=1)  # 레벨 제거
    ho_i = ho_i.reset_index()  # 인덱스 재설정
    ho_i = ho_i.set_axis(['ds', 'y'], axis=1)  # 열 이름 설정
    # ds와 y 열을 가진 데이터프레임 생성
    data_to_plot = ho_i[['ds', 'y']]
    # 'ds' 열을 날짜 형식으로 변환
    data_to_plot['ds'] = pd.to_datetime(data_to_plot['ds'])
    min_y = data_to_plot['y'].min()  # 최솟값
    max_y = data_to_plot['y'].max()  # 최댓값
    # 마지막 'ds' 날짜 확인
    last_date = data_to_plot['ds'].iloc[0]
    
    # 1년 후의 날짜 생성
    one_year_later = last_date + pd.DateOffset(years=1)
    
    date_range = pd.date_range(start=data_to_plot['ds'].min(), end=one_year_later)
    # 증강된 데이터프레임 생성할 때 날짜가 하루당 4개씩 나오게 설정
    date_range_expanded = pd.date_range(start=data_to_plot['ds'].min(), end=one_year_later, freq='6H')  # 6시간 간격으로 설정하여 4개씩 생성
    
    # 생성된 날짜로 데이터프레임 구성
    expanded_df = pd.DataFrame({'ds': date_range_expanded})
    # print(expanded_df)
    
    # 'ds' 컬럼을 기준으로 데이터프레임을 나눔
    before_date = '2023-09-01'
    test_data_to_plot_before = expanded_df[expanded_df['ds'] < before_date].copy()
    test_data_to_plot_after = expanded_df[expanded_df['ds'] >= before_date].copy()
    
    # data_to_plot 값 덮어쓰기
    test_data_to_plot_before['y'] = data_to_plot['y'].values  # y 열에 data_to_plot의 y 열 값을 넣음
    
    # 각 시간대별로 데이터 분리
    test_data_to_plot_before['hour'] = test_data_to_plot_before['ds'].dt.hour
    
    # 시간대에 따라 데이터 분리
    data_0000 = test_data_to_plot_before[test_data_to_plot_before['hour'] == 0]
    data_0600 = test_data_to_plot_before[test_data_to_plot_before['hour'] == 6]
    data_1200 = test_data_to_plot_before[test_data_to_plot_before['hour'] == 12]
    data_1800 = test_data_to_plot_before[test_data_to_plot_before['hour'] == 18]
    # 각 시간대의 최솟값과 최댓값 계산
    min_max_values_0000 = data_0000['y'].agg(['min', 'max'])
    min_max_values_0600 = data_0600['y'].agg(['min', 'max'])
    min_max_values_1200 = data_1200['y'].agg(['min', 'max'])
    min_max_values_1800 = data_1800['y'].agg(['min', 'max'])
    # 각 시간대별로 데이터 분리
    test_data_to_plot_after['hour'] = test_data_to_plot_after['ds'].dt.hour
    
    # 시간대에 따라 데이터 분리
    data_0000_after = test_data_to_plot_after[test_data_to_plot_after['hour'] == 0].copy()
    data_0600_after = test_data_to_plot_after[test_data_to_plot_after['hour'] == 6].copy()
    data_1200_after = test_data_to_plot_after[test_data_to_plot_after['hour'] == 12].copy()
    data_1800_after = test_data_to_plot_after[test_data_to_plot_after['hour'] == 18].copy()
    test_data_to_plot_after['hour'] = test_data_to_plot_after['ds'].dt.hour
    
    # 각 시간대의 최솟값과 최댓값 범위 내에서 랜덤 값 설정
    data_0000_after['y'] = np.random.uniform(min_max_values_0000['min'], min_max_values_0000['max'], len(data_0000_after))
    data_0000_after['y'] = data_0000_after['y'].map(lambda x: round(x, 2))
    data_0600_after['y'] = np.random.uniform(min_max_values_0600['min'], min_max_values_0600['max'], len(data_0600_after))
    data_0600_after['y'] = data_0600_after['y'].map(lambda x: round(x, 2))
    data_1200_after['y'] = np.random.uniform(min_max_values_1200['min'], min_max_values_1200['max'], len(data_1200_after))
    data_1200_after['y'] = data_1200_after['y'].map(lambda x: round(x, 2))
    data_1800_after['y'] = np.random.uniform(min_max_values_1800['min'], min_max_values_1800['max'], len(data_1800_after))
    data_1800_after['y'] = data_1800_after['y'].map(lambda x: round(x, 2))
    # 최종 결과를 합친 후 날짜순으로 정렬
    final_data_after = pd.concat([data_0000_after, data_0600_after, data_1200_after, data_1800_after])
    final_data_after = final_data_after.sort_values(by='ds')
    
    # 시간대 열 제거
    final_data_after = final_data_after.drop(columns='hour')
    
    combined_df = pd.concat([test_data_to_plot_before, final_data_after])
    combined_df = combined_df.drop(columns='hour')
    
    # 'ds' 값을 칼럼으로 변환하여 한 행으로 만들기
    final_data_as_row = combined_df.set_index('ds').T.reset_index(drop=True)
    final_data_as_row.reset_index(drop=True, inplace=True)
    final_data_as_row.index = (final_data_as_row.index + i+1).map(lambda x: str(x) + '호')  # 1부터 시작하는 시퀀스로 변경
    # 이번 반복에서 생성된 데이터를 all_final_data에 추가
    all_final_data = all_final_data.append(final_data_as_row)
    
# 빈 리스트 생성하여 각 호수 데이터 저장
data_for_boxplot = []

for i in range(1, 6):
    ho_data = all_final_data.loc[f'{i}호'].copy()  # 각 호수의 데이터를 복사합니다.
    ho_data = ho_data.reset_index()  # 인덱스를 열로 변환
    ho_data = ho_data.set_axis(['ds', 'y'], axis=1)  # 데이터프레임의 열 이름을 변경
    
    # y 열을 가진 데이터프레임 생성
    data_to_plot_i = ho_data[['ds', 'y']]
    
    # 모델링
    model = Prophet(yearly_seasonality=True, daily_seasonality=True)
    model.fit(ho_data)
    # 1년치 데이터 생성 (Time Shift)
    # 마지막 날짜를 확인하여 그 날짜를 기준으로 1년치 데이터 생성
    last_date = ho_data['ds'].iloc[-1]
    future = pd.DataFrame({'ds': pd.date_range(start=last_date, periods=365, freq='D')})
    # 모델을 사용하여 예측
    forecast = model.predict(future)
    
    # 예측 결과를 박스플롯에 추가
    data_for_boxplot.append(forecast['yhat'].values)
    
# 박스플롯 시각화
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(10, 6))
plt.boxplot(data_for_boxplot)
plt.xlabel('호수')
plt.ylabel('예측값')
plt.title('각 호수별 예측값의 분포')
plt.xticks(ticks=[1, 2, 3], labels=['호수 1', '호수 2', '호수 3'])
plt.tight_layout()
plt.show()

# DataFrame을 Excel 파일로 저장
# all_final_data.to_excel('전기사용량_1년.xlsx', index=True)