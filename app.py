import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score,
    f1_score, silhouette_score, classification_report
)
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Spotify — Análise de Popularidade",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

VERDE  = "#1DB954"
CORES_4 = ["#1DB954", "#FF6B6B", "#4ECDC4", "#FFD93D"]

# ============================================================
# CARREGAR E PROCESSAR DADOS
# ============================================================
@st.cache_data
def carregar_dados():
    df = pd.read_csv("dataset.csv")

    # Limpeza
    df = df.dropna(subset=["track_name","artists","track_genre","popularity"])
    df = df.drop_duplicates(subset=["track_id"], keep="first")
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["track_name","artists"], keep="first")

    if df["explicit"].dtype == object:
        df["explicit"] = df["explicit"].map({"TRUE":True,"FALSE":False,True:True,False:False})
    df["explicit"] = df["explicit"].astype(int)

    df["duration_min"] = (df["duration_ms"] / 60000).round(2)
    df = df.drop(columns=["duration_ms"])
    df = df[(df["duration_min"] >= 0.5) & (df["duration_min"] <= 15)]

    le = LabelEncoder()
    df["track_genre_encoded"] = le.fit_transform(df["track_genre"])

    df["popularity_category"] = pd.cut(
        df["popularity"],
        bins=[-1,25,50,75,100],
        labels=["Obscura","Em Ascensão","Mainstream","Hit"]
    )
    df["popularity_category_encoded"] = pd.cut(
        df["popularity"],
        bins=[-1,25,50,75,100],
        labels=[0,1,2,3]
    ).astype(int)

    return df, le

@st.cache_data
def preparar_modelos(df):
    FEATURES = [
        "danceability","energy","loudness","speechiness",
        "acousticness","instrumentalness","liveness","valence",
        "tempo","duration_min","explicit","key","mode",
        "time_signature","track_genre_encoded"
    ]

    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[["loudness","tempo","duration_min","popularity"]] = scaler.fit_transform(
        df[["loudness","tempo","duration_min","popularity"]]
    )

    X = df_norm[FEATURES]
    y_reg = df_norm["popularity"]
    y_clf = df_norm["popularity_category_encoded"]

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    _, _, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42
    )

    kmeans  = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_train)
    arvore  = DecisionTreeClassifier(max_depth=7, random_state=42).fit(X_train, y_clf_train)
    regress = LinearRegression().fit(X_train, y_reg_train)

    clusters   = kmeans.predict(X_train)
    y_pred_clf = arvore.predict(X_test)
    y_pred_reg = np.clip(regress.predict(X_test), 0, 1)

    X_train_viz = X_train.copy()
    X_train_viz["cluster"] = clusters

    return {
        "X_train": X_train, "X_test": X_test,
        "y_reg_train": y_reg_train, "y_reg_test": y_reg_test,
        "y_clf_train": y_clf_train, "y_clf_test": y_clf_test,
        "kmeans": kmeans, "arvore": arvore, "regress": regress,
        "y_pred_clf": y_pred_clf, "y_pred_reg": y_pred_reg,
        "X_train_viz": X_train_viz, "FEATURES": FEATURES
    }

# ============================================================
# CARREGAR
# ============================================================
with st.spinner("Carregando dataset e treinando modelos..."):
    df, le = carregar_dados()
    m = preparar_modelos(df)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/168px-Spotify_logo_without_text.svg.png",
    width=60
)
st.sidebar.title("🎵 Navegação")
pagina = st.sidebar.radio("Ir para:", [
    "🏠 Visão Geral",
    "📊 Análise Exploratória (EDA)",
    "🔵 K-Means Clustering",
    "🟢 Árvore de Decisão",
    "🔴 Regressão Linear",
    "📈 Comparação dos Modelos",
    "🔍 Explorador de Músicas"
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Projeto A3 — Inteligência Artificial**")
st.sidebar.markdown(f"Dataset: **{len(df):,}** músicas")
st.sidebar.markdown(f"Features: **{len(m['FEATURES'])}** atributos")

# ============================================================
# PÁGINA: VISÃO GERAL
# ============================================================
if pagina == "🏠 Visão Geral":
    st.title("🎵 O que faz uma música ser popular no Spotify?")
    st.markdown("### Projeto A3 — Inteligência Artificial | KDD Completo")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Músicas", f"{len(df):,}")
    col2.metric("Gêneros Únicos", df["track_genre"].nunique())
    col3.metric("Popularidade Média", f"{df['popularity'].mean():.1f}")
    col4.metric("Hits (pop > 75)", f"{(df['popularity']>75).sum():,}")

    st.markdown("---")
    st.markdown("### 📋 Estrutura do Projeto KDD")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Fases do KDD realizadas:**
1. ✅ **Seleção** — Dataset Spotify (API oficial)
2. ✅ **Limpeza** — Nulls, duplicatas, outliers
3. ✅ **Transformação** — Encoding, normalização
4. ✅ **Mineração** — 3 algoritmos aplicados
5. ✅ **Interpretação** — Este dashboard
        """)
    with col2:
        st.markdown("""
**Algoritmos utilizados:**
| Algoritmo | Família (Edital) | Métrica |
|---|---|---|
| K-Means | Agrupamento | Silhouette |
| Árvore de Decisão | Classificação | Acurácia |
| Regressão Linear | Regressão | R² |
        """)

    st.markdown("---")
    st.markdown("### 💡 Principais Descobertas")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🎸 **Gênero musical** é o atributo sonoro mais importante para classificar popularidade")
    with col2:
        st.warning("📉 Atributos sonoros explicam apenas **6,5%** da popularidade (R²=0.065)")
    with col3:
        st.error("🏆 **Hits são imprevisíveis** pelo som — 0% de acerto na classe Hit")

# ============================================================
# PÁGINA: EDA
# ============================================================
elif pagina == "📊 Análise Exploratória (EDA)":
    st.title("📊 Análise Exploratória de Dados")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Distribuições", "Correlações", "Por Gênero"])

    with tab1:
        st.subheader("Distribuição de Popularidade")
        fig = px.histogram(df, x="popularity", nbins=40,
            title="Distribuição de Popularidade",
            color_discrete_sequence=[VERDE], template="plotly_dark")
        fig.add_vline(x=25, line_dash="dash", line_color="#FF6B6B",
                      annotation_text="Obscura")
        fig.add_vline(x=50, line_dash="dash", line_color="#FFD93D",
                      annotation_text="Em Ascensão")
        fig.add_vline(x=75, line_dash="dash", line_color="#4ECDC4",
                      annotation_text="Mainstream")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Músicas por Categoria")
            ordem = ["Obscura","Em Ascensão","Mainstream","Hit"]
            contagem = df["popularity_category"].value_counts().reindex(ordem).reset_index()
            contagem.columns = ["Categoria","Quantidade"]
            fig2 = px.bar(contagem, x="Categoria", y="Quantidade",
                color="Categoria",
                color_discrete_map={"Obscura":"#404040","Em Ascensão":"#808080",
                                    "Mainstream":VERDE,"Hit":"#FFD700"},
                template="plotly_dark", text="Quantidade")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.subheader("Atributo vs Popularidade")
            feat_sel = st.selectbox("Selecione um atributo:",
                ["danceability","energy","acousticness","valence",
                 "loudness","tempo","speechiness","instrumentalness"])
            amostra = df.sample(3000, random_state=42)
            fig3 = px.scatter(amostra, x=feat_sel, y="popularity",
                color="popularity_category",
                color_discrete_map={"Obscura":"#404040","Em Ascensão":"#808080",
                                    "Mainstream":VERDE,"Hit":"#FFD700"},
                opacity=0.5, template="plotly_dark",
                title=f"{feat_sel} vs Popularidade")
            st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.subheader("Mapa de Correlação")
        feats_c = ["popularity","danceability","energy","loudness",
                   "valence","acousticness","tempo","speechiness",
                   "instrumentalness","liveness","explicit"]
        corr = df[feats_c].corr()
        fig_corr = px.imshow(corr, text_auto=".2f",
            color_continuous_scale="RdYlGn",
            title="Correlação entre Atributos Sonoros",
            template="plotly_dark", zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("💡 Correlação com Popularidade")
        corr_pop = corr["popularity"].drop("popularity").sort_values(ascending=False)
        fig_bar = px.bar(
            x=corr_pop.values, y=corr_pop.index, orientation="h",
            color=corr_pop.values, color_continuous_scale="RdYlGn",
            template="plotly_dark",
            labels={"x":"Correlação","y":"Atributo"},
            title="Impacto de cada atributo na popularidade"
        )
        fig_bar.add_vline(x=0, line_color="white")
        fig_bar.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("Popularidade Média por Gênero")
        top_n = st.slider("Top N gêneros:", 5, 30, 15)
        pop_genero = df.groupby("track_genre")["popularity"].agg(["mean","count"])
        pop_genero.columns = ["Popularidade Média","Quantidade"]
        pop_genero = pop_genero[pop_genero["Quantidade"] >= 50]
        pop_genero = pop_genero.sort_values("Popularidade Média", ascending=False).head(top_n)

        fig_g = px.bar(pop_genero, x=pop_genero.index, y="Popularidade Média",
            color="Popularidade Média", color_continuous_scale="Greens",
            template="plotly_dark",
            title=f"Top {top_n} Gêneros por Popularidade Média")
        fig_g.update_layout(coloraxis_showscale=False, xaxis_tickangle=-45)
        st.plotly_chart(fig_g, use_container_width=True)

# ============================================================
# PÁGINA: K-MEANS
# ============================================================
elif pagina == "🔵 K-Means Clustering":
    st.title("🔵 K-Means Clustering")
    st.markdown("**Objetivo:** Descobrir perfis sonoros naturais nas músicas do Spotify")
    st.markdown("---")

    X_train_viz = m["X_train_viz"]
    kmeans = m["kmeans"]

    sil = silhouette_score(
        m["X_train"].sample(5000, random_state=42),
        kmeans.predict(m["X_train"].sample(5000, random_state=42))
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters (K)", "4")
    col2.metric("Silhouette Score", f"{sil:.4f}")
    col3.metric("Músicas analisadas", f"{len(X_train_viz):,}")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Perfis dos Clusters", "Visualização 3D", "Distribuição"])

    with tab1:
        feats_k = ["danceability","energy","acousticness","valence",
                   "instrumentalness","tempo","loudness"]
        perfil = X_train_viz.groupby("cluster")[feats_k].mean()

        fig = px.imshow(perfil.T, text_auto=".3f",
            color_continuous_scale="RdYlGn",
            title="Perfil Sonoro Médio de Cada Cluster",
            template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
**Como interpretar:**
- Verde escuro = valor alto | Vermelho = valor baixo
- Cluster com **acousticness alta + energy baixa** → Acústico/Calmo
- Cluster com **energy alta + loudness alta** → Eletrônico/Agitado
- Cluster com **instrumentalness alta** → Instrumental/Clássico
        """)

    with tab2:
        amostra_k = X_train_viz.sample(4000, random_state=42).copy()
        amostra_k["Cluster"] = amostra_k["cluster"].map(
            {0:"Cluster 0",1:"Cluster 1",2:"Cluster 2",3:"Cluster 3"}
        )
        col_x = st.selectbox("Eixo X:", feats_k, index=0)
        col_y = st.selectbox("Eixo Y:", feats_k, index=1)
        col_z = st.selectbox("Eixo Z:", feats_k, index=2)

        fig3d = px.scatter_3d(amostra_k,
            x=col_x, y=col_y, z=col_z,
            color="Cluster", color_discrete_sequence=CORES_4,
            opacity=0.5, template="plotly_dark",
            title=f"Clusters — {col_x} × {col_y} × {col_z}")
        fig3d.update_traces(marker=dict(size=2))
        st.plotly_chart(fig3d, use_container_width=True)

    with tab3:
        dist = X_train_viz["cluster"].value_counts().sort_index()
        labels_k = [f"Cluster {i} ({v:,} músicas)" for i, v in dist.items()]
        fig_pie = px.pie(values=dist.values, names=labels_k,
            color_discrete_sequence=CORES_4,
            title="Distribuição das Músicas por Cluster",
            template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

# ============================================================
# PÁGINA: ÁRVORE DE DECISÃO
# ============================================================
elif pagina == "🟢 Árvore de Decisão":
    st.title("🟢 Árvore de Decisão")
    st.markdown("**Objetivo:** Classificar músicas por categoria de popularidade")
    st.markdown("---")

    y_clf_test = m["y_clf_test"]
    y_pred_clf = m["y_pred_clf"]
    arvore     = m["arvore"]

    acc   = accuracy_score(y_clf_test, y_pred_clf)
    prec  = precision_score(y_clf_test, y_pred_clf, average="weighted", zero_division=0)
    rec   = recall_score(y_clf_test, y_pred_clf, average="weighted", zero_division=0)
    f1    = f1_score(y_clf_test, y_pred_clf, average="weighted", zero_division=0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Acurácia", f"{acc:.1%}")
    col2.metric("Precisão", f"{prec:.4f}")
    col3.metric("Recall", f"{rec:.4f}")
    col4.metric("F1-Score", f"{f1:.4f}")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Matriz de Confusão", "Importância dos Atributos", "Por Classe"])

    with tab1:
        cm = confusion_matrix(y_clf_test, y_pred_clf)
        cm_pct = cm.astype(float) / cm.sum(axis=1)[:,np.newaxis] * 100
        rotulos = ["Obscura","Em Ascensão","Mainstream","Hit"]

        col1, col2 = st.columns(2)
        with col1:
            fig_cm = px.imshow(cm, text_auto=True,
                x=rotulos, y=rotulos,
                color_continuous_scale="Blues",
                title="Contagem Absoluta",
                template="plotly_dark")
            fig_cm.update_layout(
                xaxis_title="Previsto",
                yaxis_title="Real",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        with col2:
            fig_pct = px.imshow(cm_pct.round(1), text_auto=True,
                x=rotulos, y=rotulos,
                color_continuous_scale="Greens",
                title="Percentual por Categoria (%)",
                template="plotly_dark")
            fig_pct.update_layout(
                xaxis_title="Previsto",
                yaxis_title="Real",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_pct, use_container_width=True)

        st.warning("⚠️ **Observação:** A classe *Hit* tem 0% de acerto — o modelo nunca consegue identificar um Hit corretamente, pois os atributos sonoros não distinguem hits dos demais.")

    with tab2:
        importancias = pd.Series(
            arvore.feature_importances_, index=m["X_train"].columns
        ).sort_values(ascending=True)

        fig_imp = px.bar(
            x=importancias.values, y=importancias.index,
            orientation="h",
            color=importancias.values, color_continuous_scale="Greens",
            title="Importância de Cada Atributo (Gini)",
            template="plotly_dark",
            labels={"x":"Importância","y":""}
        )
        fig_imp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

        top = importancias.tail(1)
        st.success(f"🏆 **Atributo mais importante:** `{top.index[0]}` — {top.values[0]*100:.1f}% do poder de decisão")

    with tab3:
        prec_c  = precision_score(y_clf_test, y_pred_clf, average=None, zero_division=0)
        rec_c   = recall_score(y_clf_test, y_pred_clf, average=None, zero_division=0)
        f1_c    = f1_score(y_clf_test, y_pred_clf, average=None, zero_division=0)
        cats    = ["Obscura","Em Ascensão","Mainstream","Hit"]

        df_m = pd.DataFrame({
            "Categoria": cats*3,
            "Métrica": ["Precisão"]*4 + ["Recall"]*4 + ["F1-Score"]*4,
            "Valor": list(prec_c) + list(rec_c) + list(f1_c)
        })
        fig_cls = px.bar(df_m, x="Categoria", y="Valor",
            color="Métrica", barmode="group",
            color_discrete_sequence=[VERDE,"#FF6B6B","#4ECDC4"],
            template="plotly_dark", title="Métricas por Categoria")
        fig_cls.update_layout(yaxis_range=[0,1])
        st.plotly_chart(fig_cls, use_container_width=True)

# ============================================================
# PÁGINA: REGRESSÃO
# ============================================================
elif pagina == "🔴 Regressão Linear":
    st.title("🔴 Regressão Linear")
    st.markdown("**Objetivo:** Prever o score numérico de popularidade (0–100)")
    st.markdown("---")

    y_reg_test = m["y_reg_test"]
    y_pred_reg = m["y_pred_reg"]
    regress    = m["regress"]

    r2   = r2_score(y_reg_test, y_pred_reg)
    mae  = mean_absolute_error(y_reg_test, y_pred_reg)
    rmse = np.sqrt(np.mean((y_reg_test.values - y_pred_reg)**2))
    r2_t = r2_score(m["y_reg_train"],
                    np.clip(regress.predict(m["X_train"]), 0, 1))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² (teste)", f"{r2:.4f}", f"{r2*100:.1f}%")
    col2.metric("R² (treino)", f"{r2_t:.4f}")
    col3.metric("MAE", f"{mae:.4f}")
    col4.metric("RMSE", f"{rmse:.4f}")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Real vs Previsto", "Coeficientes", "Resíduos"])

    with tab1:
        idx = np.random.choice(len(y_reg_test), 2000, replace=False)
        real     = y_reg_test.iloc[idx].values
        previsto = y_pred_reg[idx]
        df_sc = pd.DataFrame({
            "Real": real, "Previsto": previsto,
            "Erro Absoluto": np.abs(real - previsto)
        })
        fig_sc = px.scatter(df_sc, x="Real", y="Previsto",
            color="Erro Absoluto", color_continuous_scale="RdYlGn_r",
            title=f"Real vs Previsto  (R²={r2:.4f})",
            opacity=0.5, template="plotly_dark")
        fig_sc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Predição perfeita"
        ))
        fig_sc.update_traces(marker=dict(size=4), selector=dict(mode="markers"))
        st.plotly_chart(fig_sc, use_container_width=True)

        if r2 < 0.4:
            st.error(f"🔴 R²={r2:.4f} → O som explica apenas {r2*100:.1f}% da popularidade. Fatores externos (fama, marketing) dominam.")
        elif r2 < 0.7:
            st.warning(f"🟡 R²={r2:.4f} → Explicação moderada da popularidade.")
        else:
            st.success(f"🟢 R²={r2:.4f} → Boa explicação da popularidade pelo som.")

    with tab2:
        coefs = pd.Series(regress.coef_, index=m["X_train"].columns).sort_values()
        fig_c = px.bar(
            x=coefs.values, y=coefs.index, orientation="h",
            color=coefs.values, color_continuous_scale="RdYlGn",
            title="Coeficientes — Impacto de cada atributo na popularidade",
            template="plotly_dark",
            labels={"x":"Coeficiente","y":""}
        )
        fig_c.add_vline(x=0, line_color="white", line_width=1)
        fig_c.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_c, use_container_width=True)
        st.info(
            f"🟢 Maior impacto positivo: **{coefs.idxmax()}** ({coefs.max():+.4f})\n"
            f"🔴 Maior impacto negativo: **{coefs.idxmin()}** ({coefs.min():+.4f})"
        )

    with tab3:
        residuos = y_reg_test.values - y_pred_reg
        fig_r = px.histogram(residuos, nbins=60,
            title="Distribuição dos Resíduos (ideal: centrado em 0)",
            color_discrete_sequence=["#FF6B6B"],
            template="plotly_dark",
            labels={"value":"Resíduo (Real – Previsto)"})
        fig_r.add_vline(x=0, line_color="white", line_dash="dash")
        st.plotly_chart(fig_r, use_container_width=True)

# ============================================================
# PÁGINA: COMPARAÇÃO
# ============================================================
elif pagina == "📈 Comparação dos Modelos":
    st.title("📈 Comparação dos 3 Algoritmos")
    st.markdown("---")

    y_clf_test = m["y_clf_test"]
    y_pred_clf = m["y_pred_clf"]
    y_reg_test = m["y_reg_test"]
    y_pred_reg = m["y_pred_reg"]

    acc  = accuracy_score(y_clf_test, y_pred_clf)
    prec = precision_score(y_clf_test, y_pred_clf, average="weighted", zero_division=0)
    rec  = recall_score(y_clf_test, y_pred_clf, average="weighted", zero_division=0)
    f1   = f1_score(y_clf_test, y_pred_clf, average="weighted", zero_division=0)
    r2   = r2_score(y_reg_test, y_pred_reg)
    mae  = mean_absolute_error(y_reg_test, y_pred_reg)
    sil  = silhouette_score(
        m["X_train"].sample(5000, random_state=42),
        m["kmeans"].predict(m["X_train"].sample(5000, random_state=42))
    )

    # Tabela resumo
    resumo = pd.DataFrame({
        "Algoritmo": ["K-Means","Árvore de Decisão","Regressão Linear"],
        "Família": ["Agrupamento","Classificação","Regressão"],
        "Métrica Principal": ["Silhouette Score","Acurácia","R²"],
        "Valor": [f"{sil:.4f}", f"{acc:.4f} ({acc*100:.1f}%)", f"{r2:.4f} ({r2*100:.1f}%)"],
        "Avaliação": ["Moderado (0.54)","Moderado (57%)","Baixo (6.5%)"]
    })
    st.dataframe(resumo, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de barras comparativo
        df_comp = pd.DataFrame({
            "Algoritmo": [
                "K-Means (Silhouette)",
                "Árvore (Acurácia)",
                "Árvore (F1)",
                "Regressão (R²×10)",
            ],
            "Score": [sil, acc, f1, r2*10],
            "Cor": CORES_4[:4]
        })
        fig_comp = px.bar(df_comp, x="Algoritmo", y="Score",
            color="Algoritmo", color_discrete_sequence=CORES_4,
            title="Comparação de Métricas Normalizadas",
            template="plotly_dark", text="Score")
        fig_comp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_comp.update_layout(showlegend=False, yaxis_range=[0,1])
        st.plotly_chart(fig_comp, use_container_width=True)

    with col2:
        # Radar chart
        cats_r = ["Silhouette","Acurácia","Precisão","Recall","F1","R²×10"]
        vals_r = [sil, acc, prec, rec, f1, r2*10]
        fig_rad = go.Figure(go.Scatterpolar(
            r=vals_r + [vals_r[0]],
            theta=cats_r + [cats_r[0]],
            fill="toself", fillcolor="rgba(29,185,84,0.3)",
            line=dict(color=VERDE, width=2), name="Desempenho"
        ))
        fig_rad.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            title="Radar de Desempenho", template="plotly_dark"
        )
        st.plotly_chart(fig_rad, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Análise Comparativa")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            f"**🔵 K-Means**\n"
            f"Silhouette = {sil:.4f}\n"
            f"Encontrou 4 perfis sonoros reais, mas com separação moderada — músicas tendem a ter características mistas."
        )
    with col2:
        st.success(
            f"**🟢 Árvore de Decisão**\n"
            f"Acurácia = {acc:.1%}\n"
            f"Melhor performance geral. Gênero musical é a feature decisiva (68% da importância)."
        )
    with col3:
        st.error(
            f"**🔴 Regressão Linear**\n"
            f"R² = {r2:.4f} ({r2*100:.1f}%)\n"
            f"Revela que popularidade depende principalmente de fatores NÃO sonoros."
        )

# ============================================================
# PÁGINA: EXPLORADOR
# ============================================================
elif pagina == "🔍 Explorador de Músicas":
    st.title("🔍 Explorador de Músicas")
    st.markdown("Filtre e explore as músicas do dataset")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        generos = ["Todos"] + sorted(df["track_genre"].unique().tolist())
        genero_sel = st.selectbox("Gênero:", generos)
    with col2:
        cat_sel = st.selectbox("Categoria:", ["Todas","Obscura","Em Ascensão","Mainstream","Hit"])
    with col3:
        pop_range = st.slider("Faixa de popularidade:", 0, 100, (0, 100))

    df_filtrado = df.copy()
    if genero_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado["track_genre"] == genero_sel]
    if cat_sel != "Todas":
        df_filtrado = df_filtrado[df_filtrado["popularity_category"] == cat_sel]
    df_filtrado = df_filtrado[
        (df_filtrado["popularity"] >= pop_range[0]) &
        (df_filtrado["popularity"] <= pop_range[1])
    ]

    st.markdown(f"**{len(df_filtrado):,} músicas encontradas**")

    colunas_show = ["track_name","artists","track_genre","popularity",
                    "popularity_category","danceability","energy",
                    "acousticness","valence","duration_min"]
    st.dataframe(
        df_filtrado[colunas_show].sort_values("popularity", ascending=False).head(100),
        use_container_width=True, hide_index=True
    )

    if len(df_filtrado) > 0:
        st.markdown("---")
        st.subheader("Distribuição da seleção atual")
        fig_sel = px.histogram(df_filtrado, x="popularity", nbins=20,
            color_discrete_sequence=[VERDE], template="plotly_dark",
            title=f"Popularidade — {genero_sel} | {cat_sel}")
        st.plotly_chart(fig_sel, use_container_width=True)