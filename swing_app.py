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
        
        types['marketGroupID'] = pd.to_numeric(types['marketGroupID'], errors='coerce')
        groups['marketGroupID'] = pd.to_numeric(groups['marketGroupID'], errors='coerce')
        groups['parentGroupID'] = pd.to_numeric(groups['parentGroupID'], errors='coerce')
        
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
        group_id, parent_id, name = row['marketGroupID'], row['parentGroupID'], str(row['marketGroupName'])
        node = {"label": name, "value": int(group_id)}
        
        if pd.isna(parent_id):
            root_nodes.append(node)
        else:
            if parent_id not in children_dict: children_dict[parent_id] = []
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
    """Получает текущий стакан ордеров для конкретного предмета"""
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
                st.success(f"Выбрано товаров для анализа: {len(items_to_analyze)}")
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
    # Очищаем старые результаты перед новым запуском
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
        # Сохраняем в кэш сессии
        st.session_state.swing_results = filtered_df.sort_values(by="Коридор (%)", ascending=False)

# ==========================================
# ВЫВОД РЕЗУЛЬТАТОВ И ЭТАП 2: МИКРО-АНАЛИЗ
# ==========================================
if not st.session_state.swing_results.empty:
    st.subheader(f"Топ кандидатов (Найдено: {len(st.session_state.swing_results)})")
    
    # Прячем колонку ID, чтобы не засорять экран
    display_df = st.session_state.swing_results.drop(columns=['ID'])
    
    try:
        st.dataframe(
            display_df.style.background_gradient(subset=['Тренд за период (%)'], cmap='coolwarm'),
            use_container_width=True, hide_index=True
        )
    except:
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
    st.divider()
    
    # ==========================================
    # ЭТАП 2: ГЛУБОКИЙ АНАЛИЗ СТАКАНА
    # ==========================================
    st.subheader("🔍 Микро-анализ: Проверка текущего стакана")
    st.write("Выберите товар из таблицы выше, чтобы посмотреть, выгоден ли вход на рынок прямо сейчас.")
    
    # Создаем словарь {Название: ID} для выпадающего списка
    item_dict = dict(zip(st.session_state.swing_results['Название'], st.session_state.swing_results['ID']))
    selected_item = st.selectbox("Товар для проверки:", list(item_dict.keys()))
    
    if selected_item:
        item_id = item_dict[selected_item]
        
        with st.spinner("Загрузка живых ордеров из Житы..."):
            orders_df = fetch_live_orders(JITA_REGION_ID, item_id)
            
        if not orders_df.empty:
            # Разделяем стакан на Buy и Sell
            sells = orders_df[orders_df['is_buy_order'] == False].sort_values(by='price', ascending=True)
            buys = orders_df[orders_df['is_buy_order'] == True].sort_values(by='price', ascending=False)
            
            if not sells.empty and not buys.empty:
                best_sell = sells.iloc[0]['price']
                best_buy = buys.iloc[0]['price']
                
                # Расчет реальной маржи
                gross_margin = ((best_sell - best_buy) / best_buy) * 100
                net_margin = gross_margin - total_tax_loss
                
                # Метрики колонками
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Лучший Sell (Конкуренты)", f"{best_sell:,.2f} ISK")
                col2.metric("Лучший Buy (Конкуренты)", f"{best_buy:,.2f} ISK")
                col3.metric("Грязный спред", f"{gross_margin:.2f}%")
                
                # Подсвечиваем чистую маржу красным или зеленым
                margin_color = "normal" if net_margin > 0 else "inverse"
                col4.metric("Чистая маржа (с учетом налогов)", f"{net_margin:.2f}%", delta=f"Налоги: {total_tax_loss:.2f}%", delta_color=margin_color)
                
                st.write("### Топ-5 ордеров в стакане прямо сейчас")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**🔴 Ордера на продажу (Sell)**")
                    st.dataframe(sells[['price', 'volume_remain']].head(5).rename(columns={"price": "Цена", "volume_remain": "Объем"}), hide_index=True)
                with c2:
                    st.write("**🟢 Ордера на покупку (Buy)**")
                    st.dataframe(buys[['price', 'volume_remain']].head(5).rename(columns={"price": "Цена", "volume_remain": "Объем"}), hide_index=True)
                
                if net_margin > 5:
                    st.success("✅ Рынок свободен! Текущая разница цен покрывает твои налоги с запасом. Можно выставлять ордера.")
                elif net_margin > 0:
                    st.warning("⚠️ Маржа положительная, но небольшая. Подумай, стоит ли замораживать ISK из-за такого профита.")
                else:
                    st.error("❌ Входить нельзя. Прямо сейчас конкуренты сжали стакан так сильно, что налоги съедят всю прибыль (или уведут в минус).")
            else:
                st.info("Стакан пуст (нет встречных ордеров).")
