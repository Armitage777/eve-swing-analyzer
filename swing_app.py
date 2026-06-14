import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from streamlit_tree_select import tree_select

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
        
        # --- ИСПРАВЛЕНИЕ ---
        # Принудительно конвертируем колонки с ID в числа, игнорируя текст и пустые строки ("")
        types['marketGroupID'] = pd.to_numeric(types['marketGroupID'], errors='coerce')
        groups['marketGroupID'] = pd.to_numeric(groups['marketGroupID'], errors='coerce')
        groups['parentGroupID'] = pd.to_numeric(groups['parentGroupID'], errors='coerce')
        
        # Сразу отбрасываем системные предметы, у которых нет рыночной группы (они нам не нужны)
        types = types.dropna(subset=['marketGroupID'])
        
        return groups, types
    except FileNotFoundError:
        st.error("Файлы invMarketGroups.csv и invTypes.csv не найдены! Загрузите их в папку с приложением.")
        return pd.DataFrame(), pd.DataFrame()

groups_df, types_df = load_sde_data()

# Функция рекурсивного построения дерева
@st.cache_data
def build_market_tree(groups_df):
    if groups_df.empty:
        return []
        
    children_dict = {}
    root_nodes = []
    
    for _, row in groups_df.iterrows():
        group_id = row['marketGroupID']
        parent_id = row['parentGroupID']
        name = str(row['marketGroupName'])
        
        # Гарантируем, что ID узла дерева - это стандартное целое число
        node = {"label": name, "value": int(group_id)}
        
        if pd.isna(parent_id):
            root_nodes.append(node)
        else:
            if parent_id not in children_dict:
                children_dict[parent_id] = []
            children_dict[parent_id].append(node)
            
    def attach_children(nodes):
        for node in nodes:
            node_id = node['value']
            if node_id in children_dict:
                node['children'] = children_dict[node_id]
                node['children'] = sorted(node['children'], key=lambda x: x['label'])
                attach_children(node['children'])
                
    root_nodes = sorted(root_nodes, key=lambda x: x['label'])
    attach_children(root_nodes)
    return root_nodes

# ==========================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С API И МАТЕМАТИКОЙ
# ==========================================
def fetch_market_history(region_id, type_id):
    url = f"https://esi.evetech.net/latest/markets/{region_id}/history/?datasource=tranquility&type_id={type_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return None

def analyze_item_data(df, trend_days, type_name):
    if df is None or df.empty:
        return None
        
    df['date'] = pd.to_datetime(df['date'])
    cutoff_date = datetime.utcnow() - timedelta(days=trend_days)
    df = df[df['date'] >= pd.to_datetime(cutoff_date)]
    
    if df.empty or len(df) < trend_days / 2: 
        return None

    df['daily_isk_volume'] = df['volume'] * df['average']
    avg_daily_volume_mln = df['daily_isk_volume'].mean() / 1_000_000

    price_high = df['average'].quantile(0.90)
    price_low = df['average'].quantile(0.10)
    spread_percent = ((price_high - price_low) / price_low) * 100

    x = np.arange(len(df))
    y = df['average'].values
    slope, _ = np.polyfit(x, y, 1)
    
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
# САЙДБАР: НАСТРОЙКИ И ДЕРЕВО РЫНКА
# ==========================================
st.sidebar.header("Параметры налогов")
broker_fee = st.sidebar.number_input("Комиссия брокера (%)", value=1.11, step=0.01)
sales_tax = st.sidebar.number_input("Налог с продаж (%)", value=3.37, step=0.01)

st.sidebar.header("Параметры рынка")
min_daily_volume = st.sidebar.number_input("Мин. дневной объем (млн ISK)", value=500.0, step=50.0)
min_spread_percent = st.sidebar.slider("Минимальный спред (коридор) %", min_value=1, max_value=50, value=10)
trend_days = st.sidebar.slider("Период анализа (дней)", min_value=14, max_value=90, value=30)

st.sidebar.header("Дерево товаров")

selected_type_ids = []
selected_type_names = []

if not groups_df.empty:
    nodes = build_market_tree(groups_df)
    
    with st.sidebar:
        tree_state = tree_select(nodes, no_cascade=False)
        selected_group_ids = tree_state.get('checked', [])
        
        if selected_group_ids:
            # Страховка: приводим выбранные деревом ID к числам (float), чтобы фильтр сработал идеально
            valid_ids = [float(x) for x in selected_group_ids]
            items_to_analyze = types_df[types_df['marketGroupID'].isin(valid_ids)]
            
            if not items_to_analyze.empty:
                st.success(f"Выбрано товаров для анализа: {len(items_to_analyze)}")
                if len(items_to_analyze) > 500:
                    st.warning("Внимание: Выбрано много товаров. Сбор данных из ESI займет время.")
                
                selected_type_ids = items_to_analyze['typeID'].tolist()
                selected_type_names = items_to_analyze['typeName'].tolist()
            else:
                st.warning("В выбранных группах нет конечных предметов.")
        else:
            st.info("Отметьте галочками интересующие группы товаров.")

run_analysis = st.sidebar.button("Запустить анализ через ESI", type="primary", use_container_width=True)

# ==========================================
# ОСНОВНАЯ ОБЛАСТЬ: АНАЛИЗ
# ==========================================
if run_analysis and len(selected_type_ids) > 0:
    st.subheader("Анализ выбранных товаров")
    
    progress_text = "Сбор истории продаж с серверов ESI..."
    progress_bar = st.progress(0, text=progress_text)
    
    results = []
    total_items = len(selected_type_ids)
    
    for index, (type_id, type_name) in enumerate(zip(selected_type_ids, selected_type_names)):
        percent_complete = (index + 1) / total_items
        progress_bar.progress(percent_complete, text=f"Анализ [{index+1}/{total_items}]: {type_name}")
        
        df_history = fetch_market_history(JITA_REGION_ID, type_id)
        item_metrics = analyze_item_data(df_history, trend_days, type_name)
        
        if item_metrics:
            results.append(item_metrics)
            
    progress_bar.empty()
    
    if results:
        results_df = pd.DataFrame(results)
        
        filtered_df = results_df[
            (results_df['Ср. объем (млн ISK)'] >= min_daily_volume) & 
            (results_df['Коридор (%)'] >= min_spread_percent)
        ]
        
        if not filtered_df.empty:
            st.success(f"Найдено подходящих товаров: {len(filtered_df)}")
            filtered_df = filtered_df.sort_values(by="Коридор (%)", ascending=False)
            
            st.dataframe(
                filtered_df.style.background_gradient(subset=['Тренд за период (%)'], cmap='coolwarm'),
                use_container_width=True, 
                hide_index=True
            )
            st.caption("💡 Идеальный товар имеет Высокий коридор и Тренд близкий к нулю (от -5% до 5%).")
        else:
            st.warning("Ни один предмет не прошел фильтры по объему или спреду.")
            st.write("Сырые данные до фильтрации:")
            st.dataframe(results_df, use_container_width=True, hide_index=True)
    else:
        st.error("Не удалось получить данные или в выбранных группах нет товаров с историей торгов.")
