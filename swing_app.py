import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from google import genai
from google.genai import types

# Константы
JITA_REGION_ID = 10000002
ESI_BASE_URL = "https://esi.evetech.net/latest"

# Базовый справочник товаров для анализа (ID из EVE Online)
# Ты можешь добавлять сюда любые предметы и категории
ITEMS_DICTIONARY = {
    "Wormhole Economy & Salvage": {
        "Defective Current Pump": 25592,
        "Broken Drone Transceiver": 25588,
        "Alloyed Tritanium Bar": 25600,
        "Scorched Telemetry Processor": 25597,
        "Tangled Power Conduit": 25589,
        "Conductive Polymer": 25604
    },
    "Moon Materials & Gas": {
        "Fullerite-C320": 30375,
        "Fullerite-C540": 30376,
        "Sylramic Fibers": 16670,
        "Fermionic Condensates": 16683,
        "Titanium Carbide": 16673
    }
}

# --- ФУНКЦИИ СБОРА ДАННЫХ ИЗ ESI ---

@st.cache_data(ttl=3600)
def fetch_market_history(region_id: int, type_id: int):
    """Получение 30-дневной истории торгов из ESI"""
    url = f"{ESI_BASE_URL}/markets/{region_id}/history/"
    params = {"datasource": "tranquility", "type_id": type_id}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
    except Exception:
        return None
    return None

@st.cache_data(ttl=300)
def fetch_live_orders(region_id: int, type_id: int):
    """Получение текущего стакана ордеров из ESI"""
    url = f"{ESI_BASE_URL}/markets/{region_id}/orders/"
    params = {"datasource": "tranquility", "order_type": "all", "type_id": type_id}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
    except Exception:
        return pd.DataFrame()
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
        # 1. Сбор и фильтрация истории за последние 30 дней
        df_history = fetch_market_history(JITA_REGION_ID, item_id)
        history_data = []
        if df_history is not None and not df_history.empty:
            df_history['date'] = pd.to_datetime(df_history['date'])
            cutoff = datetime.utcnow() - timedelta(days=30)
            df_hist_filtered = df_history[df_history['date'] >= pd.to_datetime(cutoff)]
            clean_history = df_hist_filtered[['date', 'average', 'volume']].copy()
            clean_history['date'] = clean_history['date'].dt.strftime('%Y-%m-%d')
            history_data = clean_history.to_dict('records')

        # 2. Сбор текущего стакана (Топ-10 лучших цен)
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

# --- ИНТЕРФЕЙС STREAMLIT ---

st.set_page_config(page_title="EVE Свинг-Аналитик Pro", layout="wide")
st.title("📊 EVE Online Свинг-Трейдинг Панель")

# Сайдбар: Настройки налогов
st.sidebar.header("Параметры налогов")
broker_fee = st.sidebar.number_input("Брокерская комиссия (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
sales_tax = st.sidebar.number_input("Налог с продаж (%)", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
total_tax_loss = broker_fee + sales_tax
st.sidebar.caption(f"Общие потери на цикл (Buy + Sell): {total_tax_loss:.2f}%")

# Сайдбар: Интеграция с ИИ
st.sidebar.divider()
st.sidebar.header("ИИ Аналитик")
api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Получите ключ в Google AI Studio")

# Основной экран: Выбор категории
st.header("1. Выбор категории для макро-анализа")
category_selected = st.selectbox("Выберите сектор рынка:", list(ITEMS_DICTIONARY.keys()))

if category_selected:
    items_to_analyze = ITEMS_DICTIONARY[category_selected]
    
    with st.spinner("Загрузка макро-данных из ESI..."):
        macro_rows = []
        for name, item_id in items_to_analyze.items():
            df_hist = fetch_market_history(JITA_REGION_ID, item_id)
            if df_hist is not None and not df_hist.empty:
                # Фильтруем за 30 дней для макро-показателей
                df_hist['date'] = pd.to_datetime(df_hist['date'])
                cutoff = datetime.utcnow() - timedelta(days=30)
                df_30 = df_hist[df_hist['date'] >= pd.to_datetime(cutoff)]
                
                if not df_30.empty:
                    avg_volume = (df_30['volume'] * df_30['average']).mean() / 1_000_000  # в млн ISK
                    min_p = df_30['average'].min()
                    max_p = df_30['average'].max()
                    corridor = ((max_p - min_p) / min_p) * 100 if min_p > 0 else 0
                    
                    # Тренд (сравнение начала и конца месяца)
                    start_price = df_30.sort_values('date').iloc[0]['average']
                    end_price = df_30.sort_values('date').iloc[-1]['average']
                    trend = ((end_price - start_price) / start_price) * 100
                    
                    macro_rows.append({
                        "ID": item_id,
                        "Название": name,
                        "Ср. объем (млн ISK)": round(avg_volume, 2),
                        "Коридор (%)": round(corridor, 2),
                        "Тренд за период (%)": round(trend, 2),
                        "Текущая цена (Ср.)": round(end_price, 2)
                    })
        
        display_df = pd.DataFrame(macro_rows)
        
    if not display_df.empty:
        st.subheader(f"Макро-статистика сектора: {category_selected}")
        
        # Добавляем колонку чекбоксов для совместимости со старым интерфейсом
        display_df.insert(0, "Выбрать", False)
        st.data_editor(display_df, use_container_width=True, disabled=["ID", "Название", "Ср. объем (млн ISK)", "Коридор (%)", "Тренд за период (%)", "Текущая цена (Ср.)"])
        
        # --- БЛОК АВТОМАТИЗАЦИИ С ИИ ---
        st.divider()
        st.subheader("🤖 Полная автоматизация (Macro -> Micro -> AI)")
        st.write("Нажмите кнопку ниже, чтобы ИИ сам выбрал лучшие товары, запросил по ним стаканы и сформировал торговый план на неделю.")
        
        if st.button("Запустить полный ИИ-цикл (Gemini 1.5 Flash)", type="primary", use_container_width=True):
            if not api_key:
                st.error("Пожалуйста, введите ваш Gemini API Key в боковой панели (раздел 'ИИ Аналитик').")
            else:
                with st.spinner("ИИ анализирует макро-данные, запрашивает глубокую историю и стаканы... Это займет около 30-60 секунд."):
                    try:
                        # 1. Инициализация клиента Gemini API
                        client = genai.Client(api_key=api_key)
                        
                        # 2. Переводим макро-таблицу в CSV формат для передачи контекста
                        macro_csv = display_df.drop(columns=['Выбрать']).to_csv(index=False)
                        
                        # 3. Создаем чат с поддержкой автоматического вызова функций
                        chat = client.chats.create(
                            model="gemini-1.5-flash",
                            config=types.GenerateContentConfig(
                                tools=[fetch_deep_market_data],
                                temperature=0.2,  # Низкая температура для минимизации галлюцинаций в цифрах
                            )
                        )
                        
                        # 4. Формируем подробный системный промпт
                        system_prompt = f"""
                        Ты — ведущий экономический советник и ИИ-аналитик рынка в EVE Online. 
                        Мой текущий инвестиционный капитал: 2 миллиарда ISK. 
                        Общие потери на налогах и комиссиях за один торговый цикл (Buy + Sell): {total_tax_loss}%.
                        
                        Моя стратегия: "Ленивый свинг-трейдинг". Я хочу выставлять крупные ордера (от 500 млн до 1 млрд ISK) и обновлять их максимум раз в неделю. Мне не важна сиюминутная конкуренция на верхушке стакана. Моя цель — поймать реальные недельные/месячные спады цены вниз (закупка) и пики вверх (распродажа).
                        
                        ВАЖНОЕ ПРАВИЛО РЫНКА: В EVE Online действует правило 4 значащих цифр (Tick Size). Изменять цену на 0.01 ISK больше нельзя. Значимыми являются только первые 4 цифры числа. Все твои финальные рекомендации по ценам ДОЛЖНЫ строго соответствовать этому правилу (например: 1,515,000 -> следующий шаг 1,516,000; или 23.45 -> 23.46). Округляй цены ордеров строго с учетом этого шага!
                        
                        Вот макро-данные по текущему сектору рынка:
                        {macro_csv}
                        
                        Твой алгоритм действий:
                        1. Изучи макро-таблицу. Выбери до 5 товаров с наилучшим сочетанием хорошего объема (чтобы переварить ордера по 500 млн) и широкого Коридора (%).
                        2. Вызови встроенную функцию `fetch_deep_market_data`, передав ей список ID выбранных товаров, чтобы получить их детальную 30-дневную историю и текущие стаканы Житы.
                        3. Получив JSON-ответ от функции, проведи математический анализ:
                           - Для цены BUY-ордера: найди стабильное историческое дно (ориентируйся на 10-й перцентиль цен, отсекая случайные единичные просадки). Проверь по стакану, чтобы цена не была зажата.
                           - Для цены SELL-ордера: найди уверенный потолок (ориентируйся на 90-й перцентиль цен).
                           - Рассчитай чистую математическую маржу с учетом налога в {total_tax_loss}%.
                        4. Сформируй красивый финальный отчет на русском языке. Для каждого товара укажи: Название, Обоснование выбора, Точную цену для BUY-ордера, Точную цену для SELL-ордера (обе цены строго по правилу 4 цифр!) и Ожидаемую чистую прибыль в % за цикл. В конце дай рекомендацию по распределению моих 2 млрд ISK.
                        """
                        
                        # 5. Отправляем запрос. SDK сам сделает вызов функции, получит данные из ESI и вернет финальный текст.
                        response = chat.send_message(system_prompt)
                        
                        st.success("✅ Анализ успешно завершен!")
                        st.markdown(response.text)
                        
                    except Exception as e:
                        st.error(f"Произошла ошибка во время работы ИИ: {e}")
                        st.info("Убедитесь, что ваш API-ключ активен и библиотека google-genai установлена правильно.")
    else:
        st.warning("Не удалось загрузить данные по товарам выбранной категории из ESI.")
