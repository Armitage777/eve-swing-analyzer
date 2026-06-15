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
            children_dict[parent_id].append(
