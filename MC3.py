#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px
import logging
import streamlit as st

# Установка фиксированного зерна для генерации случайных чисел
np.random.seed(42)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_cpl(CPL_min, costs_min, costs, a):
    """
    Расчет стоимости CPL с учетом зависимости от затрат.
    """
    h = 100000000
    # Проверка на деление на ноль
    denominator = h
    if denominator == 0:
        logging.error("Деление на ноль в функции calculate_cpl.")
        denominator = np.finfo(float).eps  # Малое число вместо нуля
    return CPL_min * (1 + a * (costs - costs_min) / denominator)

def monte_carlo_simulation(n_simulations, mix, CPL_min, Costs_min, a, ARPU,
                           C1_mean, C1_std, C2_low, C2_high, budget, channels):
    """
    Запуск симуляции Монте-Карло для расчетов метрик Revenue и RMMC.
    """
    results = []
    num_channels = len(channels)
    for _ in range(n_simulations):
        # Генерация случайных параметров
        C1 = np.random.normal(C1_mean, C1_std, num_channels)
        # Избежание деления на ноль в будущем
        C1 = np.where(C1 == 0, np.finfo(float).eps, C1)
        C2 = np.array([np.random.uniform(C2_low[i], C2_high[i]) for i in range(num_channels)])

        # Распределение бюджета
        marketing_costs = budget * mix

        # Расчет Leads и CPL
        CPL = calculate_cpl(CPL_min, Costs_min, marketing_costs, a)
        # Избежание деления на ноль
        CPL = np.where(CPL == 0, np.finfo(float).eps, CPL)
        leads = marketing_costs / CPL

        # Расчет трафика
        traffic = leads / C1

        # Расчет покупателей и дохода
        buyers = leads * C2  # Покупатели
        revenue = buyers * ARPU  # Доход с учетом ARPU

        rmmc_by_channel = revenue - marketing_costs
        # Суммарные результаты
        total_revenue = np.sum(revenue)
        total_costs = np.sum(marketing_costs)
        total_rmmc = total_revenue - total_costs  # RMMC (доход минус затраты)

        # Сохранение всех параметров в результатах симуляции
        results.append({
            'Revenue_By_Channel': revenue,
            'Total_Revenue': total_revenue,
            'Total_RMMC': total_rmmc,
            'RMMC_By_Channel': rmmc_by_channel,
            'Marketing_Costs': marketing_costs,
            'CPL': CPL,
            'Leads': leads,
            'Traffic': traffic,
            'C1': C1,
            'C2': C2,
            'ARPU': ARPU,
            'Buyers': buyers,
            'Mix_Shares': mix  # Сохраняем доли микса
        })
    return pd.DataFrame(results)

def create_summary_tables(all_results_df, ARPU, channels):
    """
    Создание сводных таблиц с использованием средних значений по каждому каналу для каждого микса.
    """
    summary_tables = {}
    num_channels = len(channels)
    for mix_label in all_results_df['Mix_Label'].unique():
        mix_results = all_results_df[all_results_df['Mix_Label'] == mix_label]
        channel_summary = []
        mix_shares = mix_results['Mix_Shares'].iloc[0]  # Получаем доли каналов в миксе
        for i in range(num_channels):
            channel_data = {
                'Channel': channels[i],
                'Mix_Share': mix_shares[i],
                'Marketing_Costs': mix_results['Marketing_Costs'].apply(lambda x: x[i]).mean(),
                'CPL': mix_results['CPL'].apply(lambda x: x[i]).mean(),
                'Leads': mix_results['Leads'].apply(lambda x: x[i]).mean(),
                'Traffic': mix_results['Traffic'].apply(lambda x: x[i]).mean(),
                'CR1': mix_results['C1'].apply(lambda x: x[i]).mean(),
                'CR2': mix_results['C2'].apply(lambda x: x[i]).mean(),
                'ARPU': ARPU,
                'Revenue':  mix_results['Revenue_By_Channel'].apply(lambda x: x[i]).mean(),
                'RMMC':  mix_results['RMMC_By_Channel'].apply(lambda x: x[i]).mean(),
                'Buyers':  mix_results['Buyers'].apply(lambda x: x[i]).mean()
            }
            channel_summary.append(channel_data)
        summary_table = pd.DataFrame(channel_summary)
        summary_tables[mix_label] = summary_table
    return summary_tables

def plot_boxplots(all_results_df):
    """
    Построение боксплотов для распределения Total_RMMC и Total_Revenue.
    """
    fig_rmmc = px.box(
        all_results_df,
        x='Mix_Label',
        y='Total_RMMC',
        title='Total RMMC Distribution for Different Mixes',
        labels={'Mix_Label': 'Mix of Channels', 'Total_RMMC': 'Total RMMC'},
        template='seaborn'
    )
    st.plotly_chart(fig_rmmc)

    fig_revenue = px.box(
        all_results_df,
        x='Mix_Label',
        y='Total_Revenue',
        title='Total Revenue Distribution for Different Mixes',
        labels={'Mix_Label': 'Mix of Channels', 'Total_Revenue': 'Total Revenue'},
        template='seaborn'
    )
    st.plotly_chart(fig_revenue)

def plot_histograms(all_results_df):
    """
    Построение гистограмм Total_RMMC для всех миксов на одном графике.
    """
    fig = px.histogram(
        all_results_df,
        x='Total_RMMC',
        color='Mix_Label',
        barmode='overlay',
        opacity=0.6,
        title='Histogram of Total RMMC for All Mixes'
    )
    st.plotly_chart(fig)

def main():
    st.title("Маркетинговая симуляция Монте-Карло")

    # Ввод таргетов
    Revenue_Target = st.number_input("Целевой доход (Revenue Target)", value=188000000)
    Marketing_Costs_Target = st.number_input("Целевые маркетинговые затраты (Marketing Costs Target)", value=0.8 * Revenue_Target)

    # Ввод параметров каналов
    st.header("Параметры каналов")

    channels = st.multiselect(
        "Выберите каналы",
        ['CPA', 'Bloggers', 'Content Marketing'],
        default=['CPA', 'Bloggers', 'Content Marketing']
    )

    # Проверка на выбор каналов
    if not channels:
        st.error("Пожалуйста, выберите хотя бы один канал.")
        return

    num_channels = len(channels)

    # Ввод параметров для каждого канала
    CPL_min = []
    Costs_min = []
    a = []
    C2_low = []
    C2_high = []

    for channel in channels:
        st.subheader(f"Параметры для {channel}")
        CPL_min.append(st.number_input(f"Минимальная стоимость лида для {channel}", value=10000))
        Costs_min.append(st.number_input(f"Минимальные затраты для {channel}", value=10000000))
        a.append(st.number_input(f"Коэффициент насыщения (a) для {channel}", value=3))
        C2_low.append(st.number_input(f"Нижняя граница C2 для {channel}", value=0.1))
        C2_high.append(st.number_input(f"Верхняя граница C2 для {channel}", value=0.2))

    CPL_min = np.array(CPL_min)
    Costs_min = np.array(Costs_min)
    a = np.array(a)
    C2_low = np.array(C2_low)
    C2_high = np.array(C2_high)

    # Ввод ARPU и параметров C1
    st.header("Общие параметры")
    ARPU = st.number_input("Средний доход с покупателя (ARPU)", value=157000)
    C1_mean = st.number_input("Среднее значение C1", value=0.04)
    C1_std = st.number_input("Стандартное отклонение C1", value=0.01)

    # Ввод миксов
    st.header("Настройка миксов бюджетов")
    mixes = []
    num_mixes = st.number_input("Количество миксов", min_value=1, max_value=10, value=3)

    for i in range(int(num_mixes)):
        st.subheader(f"Микс {i+1}")
        mix = []
        total_share = 0.0
        for channel in channels:
            share = st.slider(f"Доля для {channel} в миксе {i+1}", min_value=0.0, max_value=1.0, value=1.0/num_channels, key=f"{channel}_{i}")
            mix.append(share)
            total_share += share
        if total_share == 0:
            st.error("Сумма долей в миксе должна быть больше 0.")
            return
        # Нормализуем микс до 1
        mix = np.array(mix) / total_share
        mixes.append(mix)

    # Запуск симуляции
    if st.button("Запустить симуляцию"):
        all_results = []
        for idx, mix in enumerate(mixes):
            mix_label = f'Mix {idx + 1}'
            st.write(f"Запуск симуляции для {mix_label}")
            simulation_results = monte_carlo_simulation(
                n_simulations=1000,
                mix=mix,
                CPL_min=CPL_min,
                Costs_min=Costs_min,
                a=a,
                ARPU=ARPU,
                C1_mean=C1_mean,
                C1_std=C1_std,
                C2_low=C2_low,
                C2_high=C2_high,
                budget=Marketing_Costs_Target,
                channels=channels  # Передаём channels
            )
            simulation_results['Mix_Label'] = mix_label
            simulation_results['Mix_Shares'] = [mix] * len(simulation_results)
            all_results.append(simulation_results)
        
        # Объединение результатов
        all_results_df = pd.concat(all_results)
        
        # Создание и отображение сводных таблиц
        summary_tables = create_summary_tables(all_results_df, ARPU, channels)
        for mix_label, summary_table in summary_tables.items():
            st.subheader(f"Сводная таблица для {mix_label}")
            st.dataframe(summary_table)
        
        # Построение графиков
        st.subheader("Боксплоты Total RMMC")
        plot_boxplots(all_results_df)
        
        st.subheader("Гистограммы Total RMMC")
        plot_histograms(all_results_df)

if __name__ == "__main__":
    main()


# In[ ]:




