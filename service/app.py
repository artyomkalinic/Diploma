import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Football Analytics Backtest",
    layout="wide",
)

LEAGUE_NAMES = {
    "BL": "Бундеслига",
    "EPL": "Премьер-лига",
    "SA": "Серия А",
    "LA": "Ла Лига",
    "LG1": "Лига 1",
}

OUTCOME_LABELS = {"H": "П1", "D": "X", "A": "П2"}
PROB_COLS = {"H": "p_home", "D": "p_draw", "A": "p_away"}
ODDS_COLS = {"H": "B365H", "D": "B365D", "A": "B365A"}


def infer_model_name(filename: str) -> str:
    name = filename.lower()
    if "dc_ext" in name or "dc-ext" in name:
        return "Dixon-Coles extended"
    if "dc" in name or "dixon" in name:
        return "Dixon-Coles"
    if "xg" in name and "alpha0.0" in name:
        return "Poisson xG only"
    if "xg" in name and "alpha0.5" in name:
        return "Poisson/xG blend α=0.5"
    if "xg" in name and "alpha1.0" in name:
        return "Poisson goals only"
    if "xg" in name:
        return "Poisson/xG blend"
    if "poisson" in name:
        return "Poisson baseline"
    return Path(filename).stem


def infer_league_code(filename: str) -> str | None:
    stem = Path(filename).stem.upper()
    for code in ["EPL", "LG1", "BL", "SA", "LA"]:
        if re.search(rf"(^|_){code}($|_)", stem):
            return code
    return None


def normalize_prediction_frame(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "Home": "HomeTeam",
        "Away": "AwayTeam",
        "home_team": "HomeTeam",
        "away_team": "AwayTeam",
        "home_goals": "FTHG",
        "away_goals": "FTAG",
        "actual_result": "actual",
        "model": "model_name",
        "league": "league_code",
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "date" in df.columns:
        df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["Date"] = pd.NaT

    if "model_name" not in df.columns:
        df["model_name"] = infer_model_name(source_name)

    if "league_code" not in df.columns:
        inferred = infer_league_code(source_name)
        df["league_code"] = inferred if inferred else "UNKNOWN"

    if "league_name" not in df.columns:
        df["league_name"] = df["league_code"].map(LEAGUE_NAMES).fillna(df["league_code"])

    required = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "actual", "p_home", "p_draw", "p_away"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source_name}: нет обязательных колонок: {', '.join(missing)}")

    for c in ["FTHG", "FTAG", "p_home", "p_draw", "p_away", "B365H", "B365D", "B365A", "train_size"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "season" not in df.columns:
        y = df["Date"].dt.year
        m = df["Date"].dt.month
        start = np.where(m >= 8, y, y - 1)
        df["season"] = pd.Series(start, index=df.index).astype("Int64").astype(str) + "/" + (
            pd.Series(start, index=df.index) + 1
        ).astype("Int64").astype(str)

    df["_source_file"] = source_name
    return df


@st.cache_data(show_spinner=False)
def generate_demo_data(n_per_league_model: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    leagues = ["BL", "EPL", "SA", "LA", "LG1"]
    models = [
        ("Poisson baseline", 0.06),
        ("Dixon-Coles", 0.04),
        ("Dixon-Coles extended", 0.035),
        ("Poisson/xG blend alpha=0.5", 0.03),
    ]
    teams = {
        "BL": ["Bayern", "Dortmund", "Leipzig", "Leverkusen", "Stuttgart", "Frankfurt", "Wolfsburg", "Freiburg"],
        "EPL": ["Arsenal", "Man City", "Liverpool", "Chelsea", "Tottenham", "Newcastle", "Aston Villa", "Brighton"],
        "SA": ["Inter", "Milan", "Juventus", "Napoli", "Roma", "Lazio", "Atalanta", "Fiorentina"],
        "LA": ["Real Madrid", "Barcelona", "Atletico", "Sevilla", "Valencia", "Sociedad", "Villarreal", "Betis"],
        "LG1": ["PSG", "Marseille", "Lyon", "Monaco", "Lille", "Rennes", "Nice", "Lens"],
    }
    rows = []
    start_date = pd.Timestamp("2024-08-15")
    for league in leagues:
        strengths = {t: rng.normal(0, 0.35) for t in teams[league]}
        fixtures = []
        for i in range(n_per_league_model):
            home, away = rng.choice(teams[league], 2, replace=False)
            fixtures.append((start_date + pd.Timedelta(days=int(i * 2.4)), home, away))

        for model_name, noise in models:
            for i, (date, home, away) in enumerate(fixtures):
                h_adv = 0.22
                base_h = np.exp(0.25 + strengths[home] - 0.65 * strengths[away] + h_adv)
                base_a = np.exp(0.10 + strengths[away] - 0.65 * strengths[home])
                hg = rng.poisson(base_h)
                ag = rng.poisson(base_a)
                actual = "H" if hg > ag else "D" if hg == ag else "A"

                max_g = 8
                hp = np.array([math.exp(-base_h) * base_h**k / math.factorial(k) for k in range(max_g + 1)])
                ap = np.array([math.exp(-base_a) * base_a**k / math.factorial(k) for k in range(max_g + 1)])
                mat = np.outer(hp, ap)
                p_true = np.array([np.tril(mat, -1).sum(), np.diag(mat).sum(), np.triu(mat, 1).sum()])
                p_model = np.clip(p_true + rng.normal(0, noise, 3), 0.02, 0.92)
                p_model = p_model / p_model.sum()

                p_market = np.clip(p_true + rng.normal(0, 0.025, 3), 0.03, 0.92)
                p_market = p_market / p_market.sum()
                margin = 1.055 + rng.uniform(0, 0.025)
                odds = 1 / (p_market * margin)

                rows.append({
                    "Date": date,
                    "league_code": league,
                    "league_name": LEAGUE_NAMES[league],
                    "season": "2024/2025",
                    "model_name": model_name,
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": hg,
                    "FTAG": ag,
                    "actual": actual,
                    "p_home": p_model[0],
                    "p_draw": p_model[1],
                    "p_away": p_model[2],
                    "lambda_home": base_h,
                    "lambda_away": base_a,
                    "B365H": odds[0],
                    "B365D": odds[1],
                    "B365A": odds[2],
                    "train_size": 300 + i,
                    "_source_file": "demo_generated",
                    "_is_demo": True,
                })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    data_dir = Path("data")
    # files = sorted(list(data_dir.glob("predictions*.csv")) + list(data_dir.glob("*rolling*.csv")))
    files = sorted(data_dir.glob("predictions*.csv"))
    frames = []
    errors = []
    for file in files:
        try:
            raw = pd.read_csv(file)
            frames.append(normalize_prediction_frame(raw, file.name))
        except Exception as exc:
            errors.append(f"{file.name}: {exc}")

    if frames:
        df = pd.concat(frames, ignore_index=True)
        df["_is_demo"] = False
        return df

    return generate_demo_data()


def compute_model_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    p = df[["p_home", "p_draw", "p_away"]].clip(1e-10, 1).to_numpy()
    y = np.zeros_like(p)
    mapping = {"H": 0, "D": 1, "A": 2}
    for i, actual in enumerate(df["actual"]):
        if actual in mapping:
            y[i, mapping[actual]] = 1
    valid = y.sum(axis=1) == 1
    if valid.sum() == 0:
        return {}
    p = p[valid]
    y = y[valid]
    acc = (p.argmax(axis=1) == y.argmax(axis=1)).mean()
    logloss = -np.mean(np.sum(y * np.log(p), axis=1))
    brier = np.mean(np.sum((p - y) ** 2, axis=1))
    rps = np.mean(np.sum((np.cumsum(p[:, :2], axis=1) - np.cumsum(y[:, :2], axis=1)) ** 2, axis=1) / 2)
    return {
        "Accuracy": acc,
        "Log-Loss": logloss,
        "Brier Score": brier,
        "RPS": rps,
        "N": len(p),
    }


def add_fair_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in ODDS_COLS.values() if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = np.nan
    odds = df[["B365H", "B365D", "B365A"]].replace(0, np.nan)
    inv = 1 / odds
    total = inv.sum(axis=1)
    df["fair_home"] = inv["B365H"] / total
    df["fair_draw"] = inv["B365D"] / total
    df["fair_away"] = inv["B365A"] / total
    df["bookmaker_margin"] = total - 1
    return df


def run_betting_backtest(
    df: pd.DataFrame,
    strategy: str,
    initial_bankroll: float,
    flat_pct: float,
    kelly_fraction: float,
    ev_threshold: float,
    edge_threshold: float,
    min_odds: float,
    max_odds: float,
    max_stake_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = add_fair_probabilities(df).sort_values("Date").reset_index(drop=True)

    bankroll = float(initial_bankroll)
    peak = bankroll
    equity_rows = [{"Date": df["Date"].min(), "bankroll": bankroll, "drawdown": 0.0, "event": "start"}]
    bets = []

    for _, row in df.iterrows():
        candidates = []
        for outcome in ["H", "D", "A"]:
            p = row.get(PROB_COLS[outcome], np.nan)
            odds = row.get(ODDS_COLS[outcome], np.nan)
            fair = row.get({"H": "fair_home", "D": "fair_draw", "A": "fair_away"}[outcome], np.nan)
            if pd.isna(p) or pd.isna(odds) or pd.isna(fair) or odds <= 1:
                continue
            ev = p * odds - 1
            edge = p - fair
            if ev >= ev_threshold and edge >= edge_threshold and min_odds <= odds <= max_odds:
                candidates.append((outcome, p, fair, odds, ev, edge))

        if not candidates or bankroll <= 0:
            continue

        outcome, p, fair, odds, ev, edge = sorted(candidates, key=lambda x: x[4], reverse=True)[0]

        if strategy == "Flat":
            stake_fraction = flat_pct
        else:
            full_kelly = max((p * odds - 1) / (odds - 1), 0)
            if strategy == "Kelly":
                stake_fraction = full_kelly
            else:
                stake_fraction = kelly_fraction * full_kelly

        stake_fraction = min(max(stake_fraction, 0), max_stake_pct)
        stake = bankroll * stake_fraction
        if stake <= 0:
            continue

        won = outcome == row["actual"]
        profit = stake * (odds - 1) if won else -stake
        bankroll += profit
        peak = max(peak, bankroll)
        drawdown = (peak - bankroll) / peak if peak > 0 else 0

        bets.append({
            "Date": row["Date"],
            "league": row["league_name"],
            "model": row["model_name"],
            "match": f"{row['HomeTeam']} — {row['AwayTeam']}",
            "selected_outcome": OUTCOME_LABELS[outcome],
            "actual": OUTCOME_LABELS.get(row["actual"], row["actual"]),
            "odds": odds,
            "model_p": p,
            "fair_p": fair,
            "edge": edge,
            "EV": ev,
            "stake": stake,
            "profit": profit,
            "bankroll_after": bankroll,
            "won": won,
        })
        equity_rows.append({
            "Date": row["Date"],
            "bankroll": bankroll,
            "drawdown": drawdown,
            "event": "bet",
        })

    bets_df = pd.DataFrame(bets)
    equity_df = pd.DataFrame(equity_rows)

    if bets_df.empty:
        metrics = {
            "Ставок": 0,
            "Итоговый капитал": initial_bankroll,
            "Прибыль": 0.0,
            "ROI": 0.0,
            "Win Rate": 0.0,
            "Max Drawdown": 0.0,
            "Средний коэффициент": np.nan,
            "Средний EV": np.nan,
        }
        return bets_df, equity_df, metrics

    total_staked = bets_df["stake"].sum()
    total_profit = bets_df["profit"].sum()
    metrics = {
        "Ставок": len(bets_df),
        "Итоговый капитал": bankroll,
        "Прибыль": total_profit,
        "ROI": total_profit / total_staked if total_staked else 0.0,
        "Win Rate": bets_df["won"].mean(),
        "Max Drawdown": equity_df["drawdown"].max(),
        "Средний коэффициент": bets_df["odds"].mean(),
        "Средний EV": bets_df["EV"].mean(),
    }
    return bets_df, equity_df, metrics


def metric_card(label: str, value, fmt: str = "{:.3f}"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        st.metric(label, "—")
    elif isinstance(value, (int, np.integer)):
        st.metric(label, f"{value:,}".replace(",", " "))
    elif isinstance(value, float):
        st.metric(label, fmt.format(value))
    else:
        st.metric(label, value)


st.title("Football Analytics")

df_all = load_predictions()

if df_all.get("_is_demo", pd.Series([False])).any():
    st.warning(
        "Используйте реальные результаты ноутбуков"
    )

with st.sidebar:
    st.header("1. Данные и модель")

    league_options = sorted(df_all["league_name"].dropna().unique())
    selected_leagues = st.multiselect("Лиги", league_options, default=league_options[: min(2, len(league_options))])

    model_options = sorted(df_all["model_name"].dropna().unique())
    selected_models = st.multiselect("Модели", model_options, default=model_options[:1])

    season_options = sorted(df_all["season"].dropna().astype(str).unique())
    selected_seasons = st.multiselect("Сезоны", season_options, default=season_options)

    date_min = pd.to_datetime(df_all["Date"]).min()
    date_max = pd.to_datetime(df_all["Date"]).max()
    use_date_filter = st.checkbox("Фильтр по датам", value=False)
    if use_date_filter and pd.notna(date_min) and pd.notna(date_max):
        date_range = st.date_input("Диапазон дат", [date_min.date(), date_max.date()])
    else:
        date_range = None

    st.header("2. Ограничение выборки")
    last_n = st.slider("Последние N матчей после фильтра", 20, 2000, min(400, max(20, len(df_all))), step=20)
    if "train_size" in df_all.columns:
        min_train_size = st.number_input("Минимальный train_size", min_value=0, value=0, step=50)
    else:
        min_train_size = 0

    st.header("3. Стратегия")
    strategy = st.selectbox("Стратегия", ["Flat", "Kelly", "Fractional Kelly"], index=2)
    initial_bankroll = st.number_input("Начальный банкролл", min_value=10.0, value=1000.0, step=100.0)
    flat_pct = st.slider("Flat stake, % капитала", 0.1, 10.0, 2.0, step=0.1) / 100
    kelly_fraction = st.slider("Fractional Kelly multiplier", 0.05, 1.0, 0.5, step=0.05)
    max_stake_pct = st.slider("Максимум на одну ставку, % банкролла", 0.5, 25.0, 5.0, step=0.5) / 100

    st.header("4. Value-фильтры")
    ev_threshold = st.slider("Минимальный EV", -0.10, 0.30, 0.02, step=0.005)
    edge_threshold = st.slider("Мин. преимущество p_model - p_fair", -0.10, 0.30, 0.01, step=0.005)
    min_odds, max_odds = st.slider("Диапазон коэффициентов", 1.01, 15.0, (1.30, 6.00), step=0.05)

filtered = df_all.copy()
if selected_leagues:
    filtered = filtered[filtered["league_name"].isin(selected_leagues)]
if selected_models:
    filtered = filtered[filtered["model_name"].isin(selected_models)]
if selected_seasons:
    filtered = filtered[filtered["season"].astype(str).isin(selected_seasons)]
if use_date_filter and date_range and len(date_range) == 2:
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered = filtered[(filtered["Date"] >= start) & (filtered["Date"] <= end)]
if "train_size" in filtered.columns and min_train_size > 0:
    filtered = filtered[filtered["train_size"].fillna(0) >= min_train_size]

filtered = filtered.sort_values("Date").tail(last_n).reset_index(drop=True)

tab1, tab2, tab3, tab4 = st.tabs(["Обзор", "Бэктест стратегии", "Сравнение моделей", "Матчи"])

with tab1:
    st.subheader("Выбранная выборка")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Матчей", len(filtered))
    c2.metric("Лиг", filtered["league_name"].nunique())
    c3.metric("Моделей", filtered["model_name"].nunique())
    c4.metric("Источников CSV", filtered["_source_file"].nunique())

    metrics = compute_model_metrics(filtered)
    if metrics:
        st.subheader("Вероятностные метрики на выбранной выборке")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        m2.metric("Log-Loss", f"{metrics['Log-Loss']:.3f}")
        m3.metric("Brier", f"{metrics['Brier Score']:.3f}")
        m4.metric("RPS", f"{metrics['RPS']:.3f}")
        m5.metric("N", f"{metrics['N']}")

    st.subheader("Фрагмент данных")
    show_cols = [
        "Date", "league_name", "season", "model_name", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "actual", "p_home", "p_draw", "p_away", "B365H", "B365D", "B365A",
    ]
    st.dataframe(filtered[[c for c in show_cols if c in filtered.columns]], use_container_width=True, height=360)

with tab2:
    st.subheader("Результаты стратегии")
    bets_df, equity_df, bet_metrics = run_betting_backtest(
        filtered,
        strategy=strategy,
        initial_bankroll=initial_bankroll,
        flat_pct=flat_pct,
        kelly_fraction=kelly_fraction,
        ev_threshold=ev_threshold,
        edge_threshold=edge_threshold,
        min_odds=min_odds,
        max_odds=max_odds,
        max_stake_pct=max_stake_pct,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Ставок", int(bet_metrics["Ставок"]))
    c2.metric("Итоговый капитал", f"{bet_metrics['Итоговый капитал']:.2f}")
    c3.metric("Прибыль", f"{bet_metrics['Прибыль']:.2f}")
    c4.metric("ROI", f"{bet_metrics['ROI']:.2%}")
    c5.metric("Win Rate", f"{bet_metrics['Win Rate']:.2%}")
    c6.metric("Max DD", f"{bet_metrics['Max Drawdown']:.2%}")

    if not equity_df.empty:
        fig = px.line(equity_df, x="Date", y="bankroll", title="Equity curve / динамика капитала")
        st.plotly_chart(fig, use_container_width=True)

        fig_dd = px.area(equity_df, x="Date", y="drawdown", title="Drawdown / просадка")
        fig_dd.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_dd, use_container_width=True)

    if not bets_df.empty:
        st.subheader("Список value-ставок")
        st.dataframe(
            bets_df.style.format({
                "odds": "{:.2f}",
                "model_p": "{:.3f}",
                "fair_p": "{:.3f}",
                "edge": "{:.3f}",
                "EV": "{:.3f}",
                "stake": "{:.2f}",
                "profit": "{:.2f}",
                "bankroll_after": "{:.2f}",
            }),
            use_container_width=True,
            height=420,
        )

        st.download_button(
            "Скачать ставки CSV",
            bets_df.to_csv(index=False).encode("utf-8"),
            file_name="value_bets_backtest.csv",
            mime="text/csv",
        )
    else:
        st.info("По текущим фильтрам value-ставки не найдены. Уменьшите EV/edge threshold или диапазон коэффициентов")

with tab3:
    st.subheader("Сравнение моделей")
    rows = []
    for (league, model), grp in df_all.groupby(["league_name", "model_name"]):
        m = compute_model_metrics(grp)
        if not m:
            continue
        rows.append({
            "Лига": league,
            "Модель": model,
            "Матчей": int(m["N"]),
            "Accuracy": m["Accuracy"],
            "Log-Loss": m["Log-Loss"],
            "Brier": m["Brier Score"],
            "RPS": m["RPS"],
        })
    comp = pd.DataFrame(rows)
    if not comp.empty:
        st.dataframe(
            comp.sort_values(["Лига", "Log-Loss"]).style.format({
                "Accuracy": "{:.3f}", "Log-Loss": "{:.3f}", "Brier": "{:.3f}", "RPS": "{:.3f}"
            }),
            use_container_width=True,
            height=420,
        )
        fig = px.bar(comp, x="Модель", y="Log-Loss", color="Лига", barmode="group", title="Log-Loss по моделям")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Просмотр отдельных матчей")
    if filtered.empty:
        st.info("Нет матчей в выбранной выборке.")
    else:
        tmp = filtered.copy()
        tmp["match_label"] = tmp["Date"].dt.strftime("%Y-%m-%d").fillna("") + " | " + tmp["HomeTeam"].astype(str) + " — " + tmp["AwayTeam"].astype(str) + " | " + tmp["model_name"].astype(str)
        selected_match = st.selectbox("Матч", tmp["match_label"].tolist())
        row = tmp[tmp["match_label"] == selected_match].iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("П1", f"{row['p_home']:.3f}")
        c2.metric("X", f"{row['p_draw']:.3f}")
        c3.metric("П2", f"{row['p_away']:.3f}")

        odds_cols = [c for c in ["B365H", "B365D", "B365A"] if c in row.index and pd.notna(row[c])]
        if len(odds_cols) == 3:
            fair_row = add_fair_probabilities(pd.DataFrame([row])).iloc[0]
            market_df = pd.DataFrame({
                "Исход": ["П1", "X", "П2"],
                "Модельная вероятность": [row["p_home"], row["p_draw"], row["p_away"]],
                "Честная вероятность БК": [fair_row["fair_home"], fair_row["fair_draw"], fair_row["fair_away"]],
                "Коэффициент БК": [row["B365H"], row["B365D"], row["B365A"]],
                "EV": [row["p_home"] * row["B365H"] - 1, row["p_draw"] * row["B365D"] - 1, row["p_away"] * row["B365A"] - 1],
            })
            st.dataframe(market_df.style.format({
                "Модельная вероятность БК": "{:.3f}",
                "Честная вероятность БК": "{:.3f}",
                "Коэффициент": "{:.2f}",
                "EV": "{:.3f}",
            }), use_container_width=True)

st.caption("Сервис для моделирования стратегий и анализа методов прогнозирования в спортивной аналитике")