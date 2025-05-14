import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from janome.tokenizer import Tokenizer
import numpy as np
from matplotlib import font_manager
import japanize_matplotlib

FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"

st.set_page_config(layout="wide")
st.title("パルスサーベイ 可視化ダッシュボード")

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df['年齢'] = pd.to_numeric(df['年齢'], errors='coerce')
    df['入社からの日数'] = pd.to_numeric(df['入社からの日数'], errors='coerce')
    df['回答有無'] = df['回答有無'].astype(str).str.strip().map({'あり': 1, 'なし': 0})
    df = df.dropna(subset=['年齢'])
    df['年代'] = pd.cut(df['年齢'], bins=[0, 29, 39, 49, 100], labels=['20代', '30代', '40代', '50代'])

    cond_map = {
        '好調': '好調',
        'やや好調': 'やや好調',
        '普通': '普通',
        'やや不調': 'やや不調',
        '不調': '不調'
    }
    df['コンディション'] = df['コンディション'].astype(str).str.strip().map(cond_map)
    df = df.dropna(subset=['コンディション'])

    st.subheader("数値カラムの統計量")
    st.dataframe(df.select_dtypes(include='number').describe().T)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("拠点 × 年代別 回答率")
        df_rate = df.groupby(['拠点', '年代'], dropna=False)['回答有無'].mean().reset_index()
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        sns.barplot(data=df_rate, x='拠点', y='回答有無', hue='年代',
                    palette=sns.light_palette("green", n_colors=4), ax=ax1, edgecolor='black')
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel("回答率")
        ax1.set_title("拠点 × 年代別 回答率", fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_yticklabels([f'{int(y*100)}%' for y in ax1.get_yticks()])
        for container in ax1.containers:
            labels = [f"{v.get_height() * 100:.0f}%" for v in container]
            ax1.bar_label(container, labels=labels, label_type='edge', fontsize=7)
        ax1.legend(title="年代", loc="upper right", fontsize=8)
        st.pyplot(fig1)

    with col2:
        st.subheader("拠点 × 年代別 コンディション割合")
        cond_count = df.groupby(['拠点', '年代', 'コンディション']).size().reset_index(name='件数')
        cond_count['割合'] = cond_count.groupby(['拠点', '年代'])['件数'].transform(lambda x: x / x.sum())
        cond_order = ['好調', 'やや好調', '普通', 'やや不調', '不調']
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        sns.barplot(data=cond_count, x='拠点', y='割合', hue='コンディション',
                    hue_order=cond_order, palette=sns.light_palette("green", n_colors=5),
                    ax=ax2, edgecolor='black')
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("割合")
        ax2.set_title("拠点 × 年代別 コンディション割合", fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_yticklabels([f'{int(y*100)}%' for y in ax2.get_yticks()])
        for container in ax2.containers:
            labels = [f"{v.get_height() * 100:.0f}%" for v in container]
            ax2.bar_label(container, labels=labels, label_type='edge', fontsize=7)
        ax2.legend(title="コンディション", loc="upper right", fontsize=8)
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("数値カラムの相関ヒートマップ")
        numeric_df = df.select_dtypes(include='number')
        corr = numeric_df.corr()
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        sns.heatmap(corr, annot=True, cmap='Greens', fmt=".2f", ax=ax3)
        ax3.set_title("数値カラムの相関係数", fontsize=12)
        st.pyplot(fig3)

    with col4:
        st.subheader("自由記述 ワードクラウド")
        if '自由記述' in df.columns:
            text = df['自由記述'].dropna().astype(str).str.cat(sep='。')
            tokenizer = Tokenizer()
            words = [
                token.base_form for token in tokenizer.tokenize(text)
                if token.part_of_speech.startswith("名詞")
                and token.base_form not in STOPWORDS
                and len(token.base_form) > 1
            ]
            word_string = " ".join(words)
            wordcloud = WordCloud(
                font_path=FONT_PATH,
                background_color="white",
                width=600,
                height=300,
                colormap='Greens'
            ).generate(word_string)
            fig4, ax4 = plt.subplots(figsize=(7, 4))
            ax4.imshow(wordcloud, interpolation="bilinear")
            ax4.axis("off")
            st.pyplot(fig4)

    st.subheader("カスタム集計グラフと表")

    group_col = st.selectbox("グループ軸を選択", options=["拠点", "年代", "性別", "入社からの日数"])
    target_col = st.selectbox("集計対象を選択", options=["回答有無", "コンディション"])

    if target_col == "回答有無":
        df_custom = df[[group_col, target_col]].copy()
        df_custom[target_col] = pd.to_numeric(df_custom[target_col], errors="coerce")
        df_result = df_custom.groupby(group_col)[target_col].mean().reset_index()
        df_result.rename(columns={target_col: "平均値（率）"}, inplace=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=df_result, x=group_col, y="平均値（率）",
                    palette=sns.light_palette("green", n_colors=len(df_result)), edgecolor="black", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("割合")
        ax.set_title(f"{group_col} ごとの {target_col}（平均）", fontsize=12)
        ax.set_yticklabels([f'{int(y * 100)}%' for y in ax.get_yticks()])
        for container in ax.containers:
            labels = [f"{v.get_height() * 100:.0f}%" for v in container]
            ax.bar_label(container, labels=labels, label_type='edge', fontsize=8)
        st.pyplot(fig)
        st.dataframe(df_result)

    elif target_col == "コンディション":
        df_tmp = df[[group_col, target_col]].dropna()
        cond_order = ['好調', 'やや好調', '普通', 'やや不調', '不調']
        df_count = df_tmp.groupby([group_col, target_col]).size().reset_index(name="件数")
        df_count["割合"] = df_count.groupby(group_col)["件数"].transform(lambda x: x / x.sum())

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=df_count, x=group_col, y="割合", hue=target_col,
                    hue_order=cond_order, palette=sns.light_palette("green", n_colors=5), edgecolor="black", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("割合")
        ax.set_title(f"{group_col} ごとの {target_col} 割合", fontsize=12)
        ax.set_yticklabels([f'{int(y * 100)}%' for y in ax.get_yticks()])
        for container in ax.containers:
            labels = [f"{v.get_height() * 100:.0f}%" for v in container]
            ax.bar_label(container, labels=labels, label_type='edge', fontsize=8)
        ax.legend(title=target_col, loc="upper right", fontsize=8)
        st.pyplot(fig)
        st.dataframe(df_count)