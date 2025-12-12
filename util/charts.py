from os import makedirs
from os.path import basename, join, isdir
from glob import glob

import matplotlib.pyplot as plt
from pandas import read_csv, to_datetime


def daily_chart(sheets_dir: str):
    sheets = glob(join(sheets_dir, "*.csv"))
    for sheet in sheets:
        name = basename(sheet).split(".")[0]
        df = read_csv(sheet)
        df['date'] = to_datetime(df['date'], format='%Y-%m-%d')
        ids: list = df.columns.tolist()
        ids.remove("date")

        plt.figure(figsize=(9,5.8))
        plt.figure(facecolor='white')

        for id in ids:
            plt.plot(df['date'], df[id], label=str(id), linestyle = '-', linewidth= 2, markersize=8)

        plt.grid(True)
        plt.xlabel('date', fontsize= 14)
        plt.ylabel(name, fontsize= 14)
        plt.title(name, fontweight= 'bold', fontsize= 14)
        plt.legend()

        makedirs(join(sheets_dir, "charts"), exist_ok=True)
        plt.savefig(join(sheets_dir, "charts", f'{name}.png'), bbox_inches = 'tight', pad_inches = 0.1)


def daily_sum_charts(daily_df, data,export_folder, chart_name):
    plt.figure(figsize=(9,5.8))
    plt.figure(facecolor='white')

    plt.plot(daily_df['date'], daily_df[data].cumsum(), color= 'blue', linestyle = '-',linewidth= 2, markersize=8)

    plt.xticks(daily_df['date'].dt.to_period('M').unique().to_timestamp(),
                [month.strftime('%B %Y') for month in daily_df['date'].dt.to_period('M').unique()],
                rotation=90
    )

    plt.grid(True)
    plt.xlabel('date', fontsize= 14)
    plt.ylabel(chart_name, fontsize= 14)
    plt.title(chart_name, fontweight= 'bold', fontsize= 14)

    if not isdir(export_folder):
        makedirs(export_folder)

    plt.savefig(join(export_folder, f'{chart_name}.png'), bbox_inches = 'tight', pad_inches = 0.1)


def season_ET_bar_charts(season_df, export_folder):
    # Creating bar charts for sum and mean
    plt.figure(figsize=(12, 6))
    plt.figure(facecolor='white')

    # Bar chart for sum
    plt.subplot(1, 2, 1)
    plt.bar(season_df.index, season_df['sum'], color='skyblue')
    plt.xlabel('Month', fontsize= 14)
    plt.ylabel('ETa mm', fontsize= 14)
    plt.title('Monthly ETa', fontweight= 'bold', fontsize= 14)
    plt.xticks(season_df.index, season_df['Month-Year'].dt.strftime('%B %Y'), rotation=90)

    # Bar chart for mean
    plt.subplot(1, 2, 2)
    plt.bar(season_df.index, season_df['mean'], color='salmon')
    plt.xlabel('Month', fontsize= 14)
    plt.ylabel('ETa mm/day', fontsize= 14)
    plt.title('Mean Monthly ETa', fontweight= 'bold', fontsize= 14)
    plt.xticks(season_df.index, season_df['Month-Year'].dt.strftime('%B %Y'), rotation=90)

    plt.tight_layout()

    if not isdir(export_folder):
        makedirs(export_folder)

    plt.savefig(join(export_folder, 'seasonal ETa.png'), bbox_inches = 'tight', pad_inches = 0.1)


def season_biomass_bar_charts(season_df, export_folder):
    # Creating bar charts for sum and mean
    plt.figure(figsize=(12, 6))
    plt.figure(facecolor='white')

    # Bar chart for sum
    plt.subplot(1, 2, 1)
    plt.bar(season_df.index, season_df['sum'], color='skyblue')
    plt.xlabel('Month', fontsize= 14)
    plt.ylabel('biomass kg/ha', fontsize= 14)
    plt.title('Monthly biomass', fontweight= 'bold', fontsize= 14)
    plt.xticks(season_df.index, season_df['Month-Year'].dt.strftime('%B %Y'), rotation=90)

    # Bar chart for mean
    plt.subplot(1, 2, 2)
    plt.bar(season_df.index, season_df['mean'], color='salmon')
    plt.xlabel('Month', fontsize= 14)
    plt.ylabel('biomass kg/ha/day', fontsize= 14)
    plt.title('Mean Monthly biomass', fontweight= 'bold', fontsize= 14)
    plt.xticks(season_df.index, season_df['Month-Year'].dt.strftime('%B %Y'), rotation=90)

    plt.tight_layout()

    if not isdir(export_folder):
        makedirs(export_folder)

    plt.savefig(join(export_folder, 'seasonal biomass.png'), bbox_inches = 'tight', pad_inches = 0.1)
