import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from streamlit_tree_select import tree_select

# Настройка страницы
st.set_page_config(page_title="EVE Swing Analyzer", layout="wide", page_icon="📈")
st.title("📈 EVE Online: Анализатор коридорной торговли (Жита)")

# Константы
JITA_REGION_ID = 10000002

# Инициализация кэша сессии для результатов
if 'swing_results' not in st.session_state:
    st.session_state.swing_results = pd.DataFrame()

# ==========================================
# ЗАГРУЗКА И ОБРАБОТКА SDE ФАЙЛОВ
# ==========================================
@st.cache_data
def load_sde_data():
    try:
        groups = pd.read_csv("invMarketGroups.csv")
        types = pd.read_csv("invTypes.csv")
        
        # Принудительно конвертируем колонки с ID в числа
        types['marketGroupID'] = pd.to_numeric(types['marketGroupID'], errors='coerce')
        groups['marketGroupID'] = pd.to_numeric(groups['marketGroupID'], errors='coerce')
        groups['parentGroupID'] = pd.to_numeric(groups['parentGroupID'], errors='coerce')
        
        # Отбрасываем системные предметы
        types = types.dropna(subset=['marketGroupID'])
        return groups, types
    except FileNotFoundError:
        st.error("Файлы invMarketGroups.csv и invTypes.csv не найдены! Загрузите их в папку с приложением.")
        return pd.DataFrame(), pd.DataFrame()

groups_df, types_df = load_sde_data()

@st.cache_data
def build_market_tree(groups_df):
    if groups_df.empty: return []
    children_dict = {}
    root_nodes = []
    
    for _, row in groups_df.iterrows():
        group_id = row['marketGroupID']
        parent_id = row['parentGroupID']
        name = str(row['marketGroupName'])
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
                node['children'] = sorted(children_dict[node_id], key=lambda x: x['label'])
                attach_children(node['children'])
                
    root_nodes = sorted(root_nodes, key=lambda x: x['label'])
    attach_children(root_nodes)
    return root_nodes

# ==========================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С API
# ==========================================
def fetch_market_history(region_id, type_id):
    url = f"https://esi.evetech.net/latest/markets/{region_id}/history/?datasource=tranquility&type_id={type_id}"
    response = requests.get(url)
    return pd.DataFrame(response.json()) if response.status_code == 200 else None

def fetch_live_orders(region_id, type_id):
    url = f"https://esi.evetech.net/latest/markets/{region_id}/orders/?datasource=tranquility&order_type=all&type_id={type_id}"
    response = requests.get(url)
    return pd.DataFrame(response.json()) if response.status_code == 200 else pd.DataFrame()

def analyze_item_data(df, trend_days, type_name, type_id):
    if df is None or df.empty: return None
        
    df['date'] = pd.to_datetime(df['date'])
    cutoff_date = datetime.utcnow() - timedelta(days=trend_days)
    df = df[df['date'] >= pd.to_datetime(cutoff_date)]
    
    if df.empty or len(df) < trend_days / 2: return None

    df['daily_isk_volume'] = df['volume'] * df['average']
    avg_daily_volume_mln = df['daily_isk_volume'].mean() / 1_000_000

    price_high = df['average'].quantile(0.90)
    price_low = df['average'].quantile(0.10)
    spread_percent = ((price_high - price_low) / price_low) * 100

    x = np.arange(len(df))
    y = df['average'].values
    slope, _ = np.polyfit(x, y, 1)
    total_trend_change_percent = (slope * len(df) / df['average'].mean()) * 100

    return {
        "ID": type_id,
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
total_tax_loss = (broker_fee * 2) + sales_tax

st.sidebar.header("Параметры рынка")
min_daily_volume = st.sidebar.number_input("Мин. дневной объем (млн ISK)", value=500.0, step=50.0)
min_spread_percent = st.sidebar.slider("Минимальный спред (коридор) %", min_value=1, max_value=50, value=10)
trend_days = st.sidebar.slider("Период анализа (дней)", min_value=14, max_value=90, value=30)

st.sidebar.header("Дерево товаров")
selected_type_ids, selected_type_names = [], []

if not groups_df.empty:
    nodes = build_market_tree(groups_df)
    with st.sidebar:
        tree_state = tree_select(nodes, no_cascade=False)
        selected_group_ids = tree_state.get('checked', [])
        
        if selected_group_ids:
            valid_ids = [float(x) for x in selected_group_ids]
            items_to_analyze = types_df[types_df['marketGroupID'].isin(valid_ids)]
            
            if not items_to_analyze.empty:
                st.success(f"Выбрано товаров: {len(items_to_analyze)}")
                selected_type_ids = items_to_analyze['typeID'].tolist()
                selected_type_names = items_to_analyze['typeName'].tolist()
            else:
                st.warning("В выбранных группах нет конечных предметов.")
        else:
            st.info("Отметьте галочками интересующие группы товаров.")

run_analysis = st.sidebar.button("Запустить макро-анализ (ESI)", type="primary", use_container_width=True)

# ==========================================
# ЭТАП 1: МАКРО-АНАЛИЗ (ИСТОРИЯ)
# ==========================================
if run_analysis and len(selected_type_ids) > 0:
    st.session_state.swing_results = pd.DataFrame()
    
    progress_bar = st.progress(0, text="Сбор истории продаж...")
    results = []
    
    for index, (type_id, type_name) in enumerate(zip(selected_type_ids, selected_type_names)):
        progress_bar.progress((index + 1) / len(selected_type_ids), text=f"Анализ: {type_name}")
        df_history = fetch_market_history(JITA_REGION_ID, type_id)
        metrics = analyze_item_data(df_history, trend_days, type_name, type_id)
        if metrics: results.append(metrics)
            
    progress_bar.empty()
    
    if results:
        df_res = pd.DataFrame(results)
        filtered_df = df_res[
            (df_res['Ср. объем (млн ISK)'] >= min_daily_volume) & 
            (df_res['Коридор (%)'] >= min_spread_percent)
        ]
        st.session_state.swing_results = filtered_df.sort_values(by="Коридор (%)", ascending=False)

# ==========================================
# ВЫВОД РЕЗУЛЬТАТОВ И ЭТАП 2: МИКРО-АНАЛИЗ
# ==========================================
if not st.session_state.swing_results.empty:
    st.subheader(f"Топ кандидатов (Найдено: {len(st.session_state.swing_results)})")
    
    # Кнопка скачивания макро-отчета
    csv_macro = st.session_state.swing_results.drop(columns=['ID']).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Скачать таблицу макроанализа (CSV)",
        data=csv_macro,
        file_name=f"macro_analysis_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv",
        mime="text/csv",
    )
    
    st.divider()
    
    # Интерактивная таблица для выбора
    st.subheader("✅ Отбор для глубокого ИИ-анализа")
    st.write("Отметьте галочками товары для выгрузки истории и стакана, затем нажмите кнопку ниже.")
    
    display_df = st.session_state.swing_results.copy()
    if "Выбрать" not in display_df.columns:
        display_df.insert(0, "Выбрать", False)
        
    edited_df = st.data_editor(
        display_df.drop(columns=['ID']),
        hide_index=True,
        column_config={
            "Выбрать": st.column_config.CheckboxColumn("Анализировать", default=False)
        },
        disabled=[col for col in display_df.columns if col not in ["Выбрать", "ID"]], 
        use_container_width=True
    )
    
    selected_names = edited_df[edited_df["Выбрать"] == True]["Название"].tolist()
    selected_rows = display_df[display_df["Название"].isin(selected_names)]

    # Сбор данных для JSON
    if not selected_rows.empty:
        if st.button("📥 Собрать полные данные для ИИ-аналитика", type="primary"):
            progress_text = "Сбор истории и стаканов из Житы..."
            micro_progress = st.progress(0, text=progress_text)
            
            deep_analysis_data = []
            total_selected = len(selected_rows)
            
            for i, (_, row) in enumerate(selected_rows.iterrows()):
                micro_progress.progress((i + 1) / total_selected, text=f"Запрос данных: {row['Название']}")
                
                item_id = row['ID']
                item_name = row['Название']
                
                # 1. Запрос истории
                df_history = fetch_market_history(JITA_REGION_ID, item_id)
                history_data = []
                if df_history is not None and not df_history.empty:
                    df_history['date'] = pd.to_datetime(df_history['date'])
                    cutoff_date = datetime.utcnow() - timedelta(days=trend_days)
                    df_hist_filtered = df_history[df_history['date'] >= pd.to_datetime(cutoff_date)]
                    
                    clean_history = df_hist_filtered[['date', 'average', 'volume']].copy()
                    clean_history['date'] = clean_history['date'].dt.strftime('%Y-%m-%d')
                    history_data = clean_history.to_dict('records')

                # 2. Запрос стакана
                orders_df = fetch_live_orders(JITA_REGION_ID, item_id)
                buy_orders = []
                sell_orders = []
                if not orders_df.empty:
                    sells = orders_df[orders_df['is_buy_order'] == False].sort_values(by='price', ascending=True).head(10)
                    buys = orders_df[orders_df['is_buy_order'] == True].sort_values(by='price', ascending=False).head(10)
                    
                    sell_orders = sells[['price', 'volume_remain']].to_dict('records')
                    buy_orders = buys[['price', 'volume_remain']].to_dict('records')

                # 3. Упаковка
                deep_analysis_data.append({
                    "Item_Name": item_name,
                    "Item_ID": int(item_id),
                    "Macro_Trend_Percent": float(row['Тренд за период (%)']),
                    "History": history_data,
                    "Live_Order_Book": {
                        "Top_10_Sells": sell_orders,
                        "Top_10_Buys": buy_orders
                    }
                })
                    
            micro_progress.empty()
            
            if deep_analysis_data:
                st.success("✅ Сбор глубоких данных завершен! Скачайте JSON-файл и отправьте его мне в чат.")
                
                json_str = json.dumps(deep_analysis_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="⬇️ Скачать файл для ИИ-анализа (JSON)",
                    data=json_str.encode('utf-8'),
                    file_name=f"deep_analysis_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.json",
                    mime="application/json",
                    type="primary"
                )
