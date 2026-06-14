import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="EVE Swing Analyzer", layout="wide", page_icon="📈")
st.title("📈 EVE Online: Анализатор коридорной торговли (Жита)")

# Константы
JITA_REGION_ID = 10000002

# ==========================================
# ЗАГРУЗКА И ОБРАБОТКА SDE ФАЙЛОВ
# ==========================================
@st.cache_data
def load_sde_data():
    try:
        groups = pd.read_csv("invMarketGroups.csv")
        types = pd.read_csv("invTypes.csv")
        groups['parentGroupID'] = pd.to_numeric(groups['parentGroupID'], errors='coerce')
        return groups, types
    except FileNotFoundError:
        st.error("Файлы invMarketGroups.csv и invTypes.csv не найдены! Загрузите их в папку с приложением.")
        return pd.DataFrame(), pd.DataFrame()

groups_df, types_df = load_sde_data()

# ==========================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С API И МАТЕМАТИКОЙ
# ==========================================
def fetch_market_history(region_id, type_id):
    """Скачивает историю рынка для конкретного предмета"""
    url = f"https://esi.evetech.net/latest/markets/{region_id}/history/?datasource=tranquility&type_id={type_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return None

def analyze_item_data(df, trend_days, type_name):
    """Анализирует DataFrame с историей и возвращает метрики"""
    if df is None or df.empty:
        return None
        
    # Конвертируем даты и фильтруем по периоду
    df['date'] = pd.to_datetime(df['date'])
    cutoff_date = datetime.utcnow() - timedelta(days=trend_days)
    df = df[df['date'] >= pd.to_datetime(cutoff_date)]
    
    if df.empty or len(df) < trend_days / 2: # Если данных мало (например, предмет почти не продается)
        return None

    # 1. Расчет объема (в млн ISK)
    df['daily_isk_volume'] = df['volume'] * df['average']
    avg_daily_volume_mln = df['daily_isk_volume'].mean() / 1_000_000

    # 2. Расчет спреда (коридора)
    # Используем 90-й и 10-й перцентили цен, чтобы отбросить случайные спайки
    price_high = df['average'].quantile(0.90)
    price_low = df['average'].quantile(0.10)
    spread_percent = ((price_high - price_low) / price_low) * 100

    # 3. Расчет тренда (Линейная регрессия)
    # Смотрим, куда глобально движется цена
    x = np.arange(len(df))
    y = df['average'].values
    slope, _ = np.polyfit(x, y, 1)
    
    # Выражаем наклон в процентах от средней цены за весь период
    mean_price = df['average'].mean()
    total_trend_change_percent = (slope * len(df) / mean_price) * 100

    return {
        "Название": type_name,
        "Ср. объем (млн ISK)": round(avg_daily_volume_mln, 2),
        "Коридор (%)": round(spread_percent, 2),
        "Тренд за период (%)": round(total_trend_change_percent, 2),
        "Текущая цена (Ср.)": round(df['average'].iloc[-1], 2)
    }

# ==========================================
# САЙДБАР: НАСТРОЙКИ
# ==========================================
st.sidebar.header("Параметры налогов")
broker_fee = st.sidebar.number_input("Комиссия брокера (%)", value=1.11, step=0.01)
sales_tax = st.sidebar.number_input("Налог с продаж (%)", value=3.37, step=0.01)

st.sidebar.header("Параметры рынка")
min_daily_volume = st.sidebar.number_input("Мин. дневной объем (млн ISK)", value=500.0, step=50.0)
min_spread_percent = st.sidebar.slider("Минимальный спред (коридор) %", min_value=1, max_value=50, value=10)
trend_days = st.sidebar.slider("Период анализа (дней)", min_value=14, max_value=90, value=30)

# ==========================================
# САЙДБАР: ДИНАМИЧЕСКОЕ ДЕРЕВО РЫНКА
# ==========================================
st.sidebar.header("Дерево товаров")

current_parent_id = np.nan
selected_group_id = None
items_to_analyze = pd.DataFrame()

if not groups_df.empty:
    for level in range(1, 7):
        if pd.isna(current_parent_id):
            options = groups_df[groups_df['parentGroupID'].isna()]
        else:
            options = groups_df[groups_df['parentGroupID'] == current_parent_id]
            
        if options.empty:
            break

        option_dict = dict(zip(options['marketGroupName'], options['marketGroupID']))
        selected_name = st.sidebar.selectbox(f"Уровень {level}:", list(option_dict.keys()), key=f"lvl_{level}")
        current_parent_id = option_dict[selected_name]
        selected_group_id = current_parent_id
        
        has_types = groups_df[groups_df['marketGroupID'] == current_parent_id]['hasTypes'].iloc[0]
        if has_types == 1:
            items_to_analyze = types_df[types_df['marketGroupID'] == selected_group_id]
            break

    if not items_to_analyze.empty:
        st.sidebar.info(f"Предметов для анализа: {len(items_to_analyze)}")
        # Ограничиваем на всякий случай, чтобы случайно не повесить сервер выбрав слишком большую группу
        if len(items_to_analyze) > 500:
            st.sidebar.warning("Внимание: Группа очень большая. Анализ может занять много времени.")
            
        selected_type_ids = items_to_analyze['typeID'].tolist()
        selected_type_names = items_to_analyze['typeName'].tolist()
    else:
        st.sidebar.warning("Выберите конечную группу товаров.")
        selected_type_ids = []

run_analysis = st.sidebar.button("Запустить анализ через ESI", type="primary", use_container_width=True)

# ==========================================
# ОСНОВНАЯ ОБЛАСТЬ: АНАЛИЗ
# ==========================================
if run_analysis and len(selected_type_ids) > 0:
    st.subheader(f"Анализ группы: {selected_name}")
    
    progress_text = "Подключение к ESI API..."
    progress_bar = st.progress(0, text=progress_text)
    
    results = []
    total_items = len(selected_type_ids)
    
    # Перебираем все предметы в группе
    for index, (type_id, type_name) in enumerate(zip(selected_type_ids, selected_type_names)):
        # Обновляем прогресс-бар
        percent_complete = (index + 1) / total_items
        progress_bar.progress(percent_complete, text=f"Анализ [{index+1}/{total_items}]: {type_name}")
        
        # Запрашиваем и анализируем
        df_history = fetch_market_history(JITA_REGION_ID, type_id)
        item_metrics = analyze_item_data(df_history, trend_days, type_name)
        
        if item_metrics:
            results.append(item_metrics)
            
    progress_bar.empty() # Убираем прогресс-бар после завершения
    
    # Обрабатываем результаты
    if results:
        results_df = pd.DataFrame(results)
        
        # Применяем фильтры пользователя
        filtered_df = results_df[
            (results_df['Ср. объем (млн ISK)'] >= min_daily_volume) & 
            (results_df['Коридор (%)'] >= min_spread_percent)
        ]
        
        if not filtered_df.empty:
            st.success(f"Найдено подходящих товаров: {len(filtered_df)}")
            
            # Сортируем по умолчанию по ширине коридора
            filtered_df = filtered_df.sort_values(by="Коридор (%)", ascending=False)
            
            # Подсвечиваем колонку с трендом для удобства
            st.dataframe(
                filtered_df.style.background_gradient(subset=['Тренд за период (%)'], cmap='coolwarm'),
                use_container_width=True, 
                hide_index=True
            )
            
            st.caption("💡 Подсказка: Идеальный товар для коридорной торговли имеет **Высокий коридор** и **Тренд близкий к нулю** (от -5% до 5%). Сильно красные или сильно синие значения в Тренде означают, что товар не колеблется, а направленно растет или падает.")
        else:
            st.warning("Ни один предмет не прошел фильтры. Попробуйте снизить требования к объему или спреду.")
            st.write("Сырые данные до фильтрации (для справки):")
            st.dataframe(results_df, use_container_width=True, hide_index=True)
    else:
        st.error("Не удалось получить данные или в выбранной группе нет товаров с историей торгов.")
