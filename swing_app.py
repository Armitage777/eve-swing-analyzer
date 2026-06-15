import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from streamlit_tree_select import tree_select
from google import genai
from google.genai import types

# Настройка страницы
st.set_page_config(page_title="EVE Свинг-Аналитик Pro", layout="wide", page_icon="📈")
st.title("📊 EVE Online: ИИ-Свинг Аналитик (Жита)")

# Константы
JITA_REGION_ID = 10000002
ESI_BASE_URL = "https://esi.evetech.net/latest"

# Инициализация кэша сессии
if 'macro_results' not in st.session_state:
    st.session_state.macro_results = pd.DataFrame()

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
# ФУНКЦИИ СБОРА ДАННЫХ ИЗ ESI
# ==========================================
def fetch_market_history(region_id: int, type_id: int):
    url = f"{ESI_BASE_URL}/markets/{region_id}/history/"
    params = {"datasource": "tranquility", "type_id": type_id}
    try:
        response = requests.get(url, params=params, timeout=10)
        return pd.DataFrame(response.json()) if response.status_code == 200 else None
    except Exception:
        return None

def fetch_live_orders(region_id: int, type_id: int):
    url = f"{ESI_BASE_URL}/markets/{region_id}/orders/"
    params = {"datasource": "tranquility", "order_type": "all", "type_id": type_id}
    try:
        response = requests.get(url, params=params, timeout=10)
        return pd.DataFrame(response.json()) if response.status_code == 200 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# --- ФУНКЦИЯ-ИНСТРУМЕНТ (TOOL) ДЛЯ ИИ ---
def fetch_deep_market_data(item_ids: list[int]) -> str:
    """
    Fetches detailed market history (last 30 days) and current live order books 
    (Top 10 buy/sell orders) for a specific list of EVE Online item IDs.
    
    Args:
        item_ids: A list of integer Item IDs to analyze.
        
    Returns:
        A JSON string containing the deep market data.
    """
    deep_analysis_data = []
    
    for item_id in item_ids:
        # 1. История за 30 дней
        df_history = fetch_market_history(JITA_REGION_ID, item_id)
        history_data = []
        if df_history is not None and not df_history.empty:
            df_history['date'] = pd.to_datetime(df_history['date'])
            cutoff = datetime.utcnow() - timedelta(days=30)
            df_hist_filtered = df_history[df_history['date'] >= pd.to_datetime(cutoff)]
            clean_history = df_hist_filtered[['date', 'average', 'volume']].copy()
            clean_history['date'] = clean_history['date'].dt.strftime('%Y-%m-%d')
            history_data = clean_history.to_dict('records')

        # 2. Текущий стакан
        orders_df = fetch_live_orders(JITA_REGION_ID, item_id)
        buy_orders, sell_orders = [], []
        if not orders_df.empty:
            sells = orders_df[orders_df['is_buy_order'] == False].sort_values(by='price', ascending=True).head(10)
            buys = orders_df[orders_df['is_buy_order'] == True].sort_values(by='price', ascending=False).head(10)
            sell_orders = sells[['price', 'volume_remain']].to_dict('records')
            buy_orders = buys[['price', 'volume_remain']].to_dict('records')

        deep_analysis_data.append({
            "Item_ID": int(item_id),
            "History": history_data,
            "Live_Order_Book": {
                "Top_10_Sells": sell_orders,
                "Top_10_Buys": buy_orders
            }
        })
        
    return json.dumps(deep_analysis_data)

# ==========================================
# САЙДБАР: НАСТРОЙКИ, КЛЮЧ И ДЕРЕВО
# ==========================================
st.sidebar.header("Параметры налогов")
broker_fee = st.sidebar.number_input("Брокерская комиссия (%)", min_value=0.0, max_value=10.0, value=1.11, step=0.01)
sales_tax = st.sidebar.number_input("Налог с продаж (%)", min_value=0.0, max_value=10.0, value=3.37, step=0.01)
total_tax_loss = (broker_fee * 2) + sales_tax
st.sidebar.caption(f"Общие потери на цикл (Buy + Sell): {total_tax_loss:.2f}%")

st.sidebar.divider()
st.sidebar.header("ИИ Аналитик")
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Введите ваш ключ для авто-анализа")

st.sidebar.divider()
st.sidebar.header("Фильтры Макро-анализа")
min_daily_volume = st.sidebar.number_input("Мин. дневной объем (млн ISK)", value=500.0, step=50.0)
min_spread_percent = st.sidebar.slider("Минимальный спред (коридор) %", min_value=1, max_value=50, value=10)

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
            st.info("Отметьте группы товаров.")

run_macro = st.sidebar.button("1. Запустить макро-анализ", type="primary", use_container_width=True)

# ==========================================
# ЭТАП 1: СБОР МАКРО-ДАННЫХ
# ==========================================
if run_macro and len(selected_type_ids) > 0:
    st.session_state.macro_results = pd.DataFrame()
    progress_bar = st.progress(0, text="Сбор истории продаж...")
    results = []
    
    for index, (type_id, type_name) in enumerate(zip(selected_type_ids, selected_type_names)):
        progress_bar.progress((index + 1) / len(selected_type_ids), text=f"Анализ: {type_name}")
        df_hist = fetch_market_history(JITA_REGION_ID, type_id)
        
        if df_hist is not None and not df_hist.empty:
            df_hist['date'] = pd.to_datetime(df_hist['date'])
            cutoff = datetime.utcnow() - timedelta(days=30)
            df_30 = df_hist[df_hist['date'] >= pd.to_datetime(cutoff)]
            
            if not df_30.empty:
                avg_volume = (df_30['volume'] * df_30['average']).mean() / 1_000_000
                min_p = df_30['average'].quantile(0.10) # 10-й перцентиль для коридора
                max_p = df_30['average'].quantile(0.90) # 90-й перцентиль для коридора
                corridor = ((max_p - min_p) / min_p) * 100 if min_p > 0 else 0
                
                # Тренд (сравнение начала и конца периода)
                start_price = df_30.sort_values('date').iloc[0]['average']
                end_price = df_30.sort_values('date').iloc[-1]['average']
                trend = ((end_price - start_price) / start_price) * 100 if start_price > 0 else 0
                
                results.append({
                    "ID": type_id,
                    "Название": type_name,
                    "Ср. объем (млн ISK)": round(avg_volume, 2),
                    "Коридор (%)": round(corridor, 2),
                    "Тренд за период (%)": round(trend, 2),
                    "Текущая цена (Ср.)": round(end_price, 2)
                })
                
    progress_bar.empty()
    
    if results:
        df_res = pd.DataFrame(results)
        filtered_df = df_res[
            (df_res['Ср. объем (млн ISK)'] >= min_daily_volume) & 
            (df_res['Коридор (%)'] >= min_spread_percent)
        ]
        st.session_state.macro_results = filtered_df.sort_values(by="Коридор (%)", ascending=False)

# ==========================================
# ВЫВОД ТАБЛИЦЫ И ЭТАП 2: ИИ-АВТОМАТИЗАЦИЯ
# ==========================================
if not st.session_state.macro_results.empty:
    st.subheader(f"Кандидаты макро-анализа (Найдено: {len(st.session_state.macro_results)})")
    
    # Показываем таблицу
    display_df = st.session_state.macro_results.copy()
    try:
        st.dataframe(display_df.style.background_gradient(subset=['Тренд за период (%)'], cmap='coolwarm'), use_container_width=True, hide_index=True)
    except:
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
    st.divider()
    
    # БЛОК ИИ
    st.subheader("🤖 Этап 2: Полный ИИ-анализ стаканов")
    st.write("Нажмите кнопку ниже, чтобы ИИ сам выбрал 5 лучших товаров из таблицы выше, запросил их стаканы и сформировал торговые ордера.")
    
    if st.button("2. Запустить ИИ-цикл (Gemini 1.5 Flash)", type="primary", use_container_width=True):
        if not api_key:
            st.error("❌ Пожалуйста, введите ваш Gemini API Key в боковой панели слева.")
        else:
            with st.spinner("ИИ анализирует макро-данные, делает запросы в ESI и считает уровни ордеров..."):
                try:
                    # Инициализация клиента Gemini API
                    client = genai.Client(api_key=api_key)
                    
                    # Переводим макро-таблицу в CSV
                    macro_csv = display_df.to_csv(index=False)
                    
                    # Настраиваем чат с функцией
                    chat = client.chats.create(
                        model="gemini-1.5-flash",
                        config=types.GenerateContentConfig(
                            tools=[fetch_deep_market_data],
                            temperature=0.2, 
                        )
                    )
                    
                    # Формируем промпт
                    system_prompt = f"""
                    Ты — экономический ИИ-аналитик EVE Online. 
                    Капитал пользователя: 2 миллиарда ISK. 
                    Налоги и брокерские комиссии (на полный цикл): {total_tax_loss}%.
                    Стратегия: "Ленивый свинг" (ордера по 500+ млн, обновление раз в неделю).
                    ВАЖНО: В EVE действует правило 4 значащих цифр (Tick Size). Шаг изменения цены зависит от порядка числа. Округляй все цены BUY и SELL строго по этому правилу!
                    
                    Макро-данные (CSV):
                    {macro_csv}
                    
                    Действия:
                    1. Выбери 5 лучших товаров (баланс объема и ширины коридора).
                    2. Вызови `fetch_deep_market_data` для этих 5 ID.
                    3. Найди 10-й перцентиль цены (дно) для BUY и 90-й перцентиль (потолок) для SELL. Оцени, не мешают ли текущие ордера в стакане.
                    4. Напиши отчет на русском с конкретными ценами покупки/продажи и ожидаемой маржой с учетом налогов.
                    """
                    
                    # Отправляем запрос (вызов функции происходит под капотом автоматически)
                    response = chat.send_message(system_prompt)
                    
                    st.success("✅ ИИ-анализ успешно завершен!")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Произошла ошибка при обращении к ИИ: {e}")
