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
        title='Распределение Total RMMC для разных миксов',
        labels={'Mix_Label': 'Микс каналов', 'Total_RMMC': 'Total RMMC'},
        template='seaborn'
    )
    st.plotly_chart(fig_rmmc)

    fig_revenue = px.box(
        all_results_df,
        x='Mix_Label',
        y='Total_Revenue',
        title='Распределение Total Revenue для разных миксов',
        labels={'Mix_Label': 'Микс каналов', 'Total_Revenue': 'Total Revenue'},
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
        title='Гистограмма Total RMMC для всех миксов'
    )
    st.plotly_chart(fig)

def main():
    st.set_page_config(page_title="Маркетинговая симуляция Монте-Карло", layout="wide")

    st.title("Маркетинговая симуляция Монте-Карло")

    # Боковая панель для ввода параметров
    with st.sidebar:
        st.header("Настройки")

        # Ввод таргетов
        Revenue_Target = st.number_input("Целевой доход (Revenue Target)", value=188000000)
        Marketing_Costs_Target = st.number_input(
            "Целевые маркетинговые затраты (Marketing Costs Target)", value=int(0.8 * Revenue_Target)
        )

        # Ввод параметров каналов с использованием экспандеров
        st.subheader("Параметры каналов")
        default_channels = ['CPA', 'Bloggers', 'Content Marketing']
        channels = default_channels  # Вы используете все каналы по умолчанию
        channel_params = {}
        for channel in channels:
            with st.expander(f"Настройки для {channel}"):
                CPL_min = st.number_input(f"Минимальная стоимость лида для {channel}", value=14000)
                Costs_min = st.number_input(f"Минимальные затраты для {channel}", value=20000000)
                a = st.number_input(f"Коэффициент насыщения (a) для {channel}", value=3)
                C2_low = st.number_input(f"Нижняя граница C2 для {channel}", value=0.18)
                C2_high = st.number_input(f"Верхняя граница C2 для {channel}", value=0.22)
                channel_params[channel] = {
                    'CPL_min': CPL_min,
                    'Costs_min': Costs_min,
                    'a': a,
                    'C2_low': C2_low,
                    'C2_high': C2_high
                }

        # Общие параметры
        st.subheader("Общие параметры")
        ARPU = st.number_input("Средний доход с покупателя (ARPU)", value=157000)
        C1_mean = st.number_input("Среднее значение C1", value=0.04)
        C1_std = st.number_input("Стандартное отклонение C1", value=0.01)

        # Настройка миксов
        st.subheader("Настройка миксов бюджетов")
        num_mixes = st.number_input("Количество миксов", min_value=1, max_value=10, value=5)
        mixes = []
        default_mixes = [
            {'CPA': 0.25, 'Bloggers': 0.25, 'Content Marketing': 0.5},
            {'CPA': 0.4, 'Bloggers': 0.4, 'Content Marketing': 0.2},
            {'CPA': 0.3, 'Bloggers': 0.3, 'Content Marketing': 0.4},
            {'CPA': 0.5, 'Bloggers': 0.3, 'Content Marketing': 0.2},
            {'CPA': 0.2, 'Bloggers': 0.5, 'Content Marketing': 0.3}
        ]
        for i in range(int(num_mixes)):
            mix = {}
            st.markdown(f"**Микс {i+1}**")
            for channel in channels:
                if i < len(default_mixes):
                    default_share = default_mixes[i][channel]
                else:
                    default_share = 1.0 / len(channels)
                share = st.number_input(f"Доля для {channel} в миксе {i+1}", min_value=0.0, max_value=1.0, value=default_share, key=f"{channel}_{i}")
                mix[channel] = share
            total_share = sum(mix.values())
            if total_share == 0:
                st.error("Сумма долей в миксе должна быть больше 0.")
                return
            # Нормализуем микс до 1
            for channel in mix:
                mix[channel] /= total_share
            mixes.append(mix)

    # Основная часть приложения
    st.header("Результаты симуляции")

    if st.button("Запустить симуляцию"):
        # Подготовка данных для симуляции
        CPL_min = np.array([channel_params[ch]['CPL_min'] for ch in channels])
        Costs_min = np.array([channel_params[ch]['Costs_min'] for ch in channels])
        a = np.array([channel_params[ch]['a'] for ch in channels])
        C2_low = np.array([channel_params[ch]['C2_low'] for ch in channels])
        C2_high = np.array([channel_params[ch]['C2_high'] for ch in channels])

        all_results = []
        for idx, mix_dict in enumerate(mixes):
            mix = np.array([mix_dict[ch] for ch in channels])
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
        
        # Отображение ключевых метрик
        average_total_rmmc = all_results_df['Total_RMMC'].mean()
        average_total_revenue = all_results_df['Total_Revenue'].mean()
        average_roi = (average_total_revenue - Marketing_Costs_Target) / Marketing_Costs_Target * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Средний Total RMMC", f"{average_total_rmmc:,.2f} руб.")
        col2.metric("Средний доход", f"{average_total_revenue:,.2f} руб.")
        col3.metric("ROI", f"{average_roi:.2f}%")

if __name__ == "__main__":
    main()