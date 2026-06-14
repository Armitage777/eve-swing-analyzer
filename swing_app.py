import streamlit as st
import pandas as pd

# Настройка страницы (широкий формат лучше подходит для таблиц и графиков)
st.set_page_config(page_title="EVE Swing Analyzer", layout="wide", page_icon="📈")

st.title("📈 EVE Online: Анализатор коридорной торговли (Жита)")

# ==========================================
# САЙДБАР: НАСТРОЙКИ И ФИЛЬТРЫ
# ==========================================
st.sidebar.header("Параметры налогов")
# Предзаполненные данные для прокачанного торговца
broker_fee = st.sidebar.number_input("Комиссия брокера (%)", value=1.11, step=0.01)
sales_tax = st.sidebar.number_input("Налог с продаж (%)", value=3.37, step=0.01)

# Считаем минимальную безубыточную разницу (грубый расчет)
total_tax_loss = (broker_fee * 2) + sales_tax
st.sidebar.caption(f"Общие потери на цикл: ~{total_tax_loss:.2f}%")

st.sidebar.header("Параметры рынка")
min_daily_volume = st.sidebar.number_input("Мин. дневной объем (млн ISK)", value=500.0, step=50.0)
min_spread_percent = st.sidebar.slider("Минимальный спред (коридор) %", min_value=1, max_value=50, value=10)
trend_days = st.sidebar.slider("Период анализа (дней)", min_value=14, max_value=90, value=30)

st.sidebar.header("Категории товаров")
# Заглушка для дерева товаров (marketGroupID). 
# Позже мы привяжем сюда реальную базу групп.
market_groups = ["Ships", "Modules", "Drones", "Minerals", "PI"]
selected_group = st.sidebar.selectbox("Выберите группу для анализа:", ["Все"] + market_groups)

st.sidebar.button("Запустить анализ", type="primary", use_container_width=True)

# ==========================================
# ОСНОВНАЯ ОБЛАСТЬ: РЕЗУЛЬТАТЫ
# ==========================================

st.subheader("Топ кандидатов для торговли")
st.write(f"Фильтры: Объем > {min_daily_volume}кк, Спред > {min_spread_percent}%, Период: {trend_days} дн.")

# Здесь позже будет функция обработки pandas dataframe
# def analyze_market_data(df, min_vol, min_spread): ...

# Создаем заглушку таблицы (mock data), чтобы увидеть, как это будет выглядеть
mock_data = pd.DataFrame({
    "Название": ["Caldari Navy Ballistic Control System", "Pithum C-Type Medium Shield Booster", "Veldspar"],
    "Ср. объем (млн/день)": [1200, 850, 15000],
    "Ширина коридора (%)": [12.5, 15.2, 5.1],
    "Текущая фаза": ["На дне (Покупать)", "Пик (Продавать)", "Стабильно"]
})

st.dataframe(mock_data, use_container_width=True, hide_index=True)

# Секция для детального графика выбранного предмета
st.subheader("Детальный график предмета")
selected_item = st.selectbox("Выберите предмет для просмотра графика:", mock_data["Название"])

st.info(f"Здесь будет выведен график цен (Bollinger Bands) за {trend_days} дней для: **{selected_item}**")
# st.line_chart(...)
