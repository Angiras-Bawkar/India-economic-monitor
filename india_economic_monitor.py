import socket
import ssl
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from requests.adapters import HTTPAdapter


# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="India Economic Monitor",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Keep the IPv4 workaround used by the original working connector.
_original_getaddrinfo = socket.getaddrinfo


def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """Resolve IPv4 addresses only (workaround for some network environments)."""
    return _original_getaddrinfo(
        host, port, socket.AF_INET, type, proto, flags
    )


socket.getaddrinfo = ipv4_only_getaddrinfo
status_placeholder = st.sidebar.empty()

class LegacySSLAdapter(HTTPAdapter):
    """HTTPS adapter preserving the TLS compatibility workaround in the original app."""

    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0x4)
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)


# ---------------------------------------------------------------------
# DARK THEME STYLING
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }
    .stApp {
        background: #0b1220;
        color: #e8eef8;
    }
    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, 0.92);
    }
    [data-testid="stSidebar"] {
        background: #101a2b;
        border-right: 1px solid #26344a;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #142238, #101a2b);
        border: 1px solid #293b54;
        padding: 16px 18px;
        border-radius: 14px;
        min-height: 120px;
    }
    [data-testid="stMetricLabel"] {
        color: #aabbd2;
    }
    [data-testid="stMetricValue"] {
        color: #f4f7fc;
    }
    .eyebrow {
        color: #79b8ff;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .75rem;
        font-weight: 700;
        margin-bottom: .35rem;
    }
    .muted {
        color: #aabbd2;
    }
    .section-card {
        background: #101a2b;
        border: 1px solid #26344a;
        border-radius: 14px;
        padding: 18px 20px;
    }
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# LIVE MoSPI DATA CONNECTOR
# ---------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_mospi_data(base_year: str = "2022-23"):
    """Fetch and clean all available quarterly GDP growth records from MoSPI."""
    url = "https://api.mospi.gov.in/api/nas/getNASData"

    params = {
        "base_year": base_year,
        "series": "Current",
        "frequency_code": "Quarterly",
        "indicator_code": "22",
        "quarterly_code": "1,2,3,4",
        "Format": "JSON",
        "page": 1,
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    try:
        with requests.Session() as session:
            session.trust_env = False
            session.mount("https://", LegacySSLAdapter())

            response = session.get(
                url, params=params, headers=headers, timeout=30
            )
            response.raise_for_status()
            payload = response.json()

            metadata = payload.get("meta_data", {})
            records = payload.get("data", [])

            if not isinstance(records, list) or not records:
                raise ValueError("MoSPI returned no GDP records on page 1.")

            total_pages = int(metadata.get("totalPages", 1))
            all_records = list(records)

            # Fetch every remaining page; the API's page size is controlled by
            # the service, so do not assume that page 1 contains all records.
            for page_number in range(2, total_pages + 1):
                page_params = {**params, "page": page_number}
                page_response = session.get(
                    url,
                    params=page_params,
                    headers=headers,
                    timeout=30,
                )
                page_response.raise_for_status()
                page_payload = page_response.json()
                page_records = page_payload.get("data", [])
                page_metadata = page_payload.get("meta_data", {})

                if not isinstance(page_records, list) or not page_records:
                    raise ValueError(
                        f"MoSPI returned no records on page {page_number}."
                    )

                returned_page = int(page_metadata.get("page", -1))
                if returned_page != page_number:
                    raise ValueError(
                        f"Requested page {page_number}, "
                        f"but API returned page {returned_page}."
                    )

                all_records.extend(page_records)

        raw = pd.DataFrame(all_records)

        required = {
            "base_year",
            "series",
            "year",
            "quarter",
            "indicator",
            "frequency",
            "current_price",
            "constant_price",
            "unit",
        }
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"Missing API fields: {sorted(missing)}")

        # Keep only the requested quarterly GDP growth series.
        raw = raw[
            (raw["base_year"].astype(str) == base_year)
            & (raw["series"] == "Current")
            & (raw["indicator"] == "GDP Growth Rate")
            & (raw["frequency"] == "Quarterly")
            & (raw["unit"] == "%")
        ].copy()

        if raw.empty:
            raise ValueError(
                f"No GDP growth records found for base year {base_year}."
            )

        raw["Real_GDP_Growth"] = pd.to_numeric(
            raw["constant_price"], errors="coerce"
        )
        raw["Nominal_GDP_Growth"] = pd.to_numeric(
            raw["current_price"], errors="coerce"
        )

        # Reference-year records can have neither growth rate available.
        both_missing = (
            raw["Real_GDP_Growth"].isna()
            & raw["Nominal_GDP_Growth"].isna()
        )
        raw = raw.loc[~both_missing].copy()

        # A single missing growth value is unexpected; do not silently plot it.
        one_missing = raw[
            ["Real_GDP_Growth", "Nominal_GDP_Growth"]
        ].isna().any(axis=1)

        if one_missing.any():
            bad_rows = raw.loc[
                one_missing,
                ["year", "quarter", "current_price", "constant_price"],
            ]
            raise ValueError(
                "Some records have only one GDP growth value missing:\n"
                + bad_rows.to_string(index=False)
            )

        if raw.empty:
            raise ValueError(
                f"No valid GDP growth observations for base year {base_year}."
            )

        if raw[
            ["Real_GDP_Growth", "Nominal_GDP_Growth"]
        ].isna().any().any():
            raise ValueError("Some GDP growth records contain invalid numeric values.")

        raw["year"] = raw["year"].astype(str)
        raw["quarter"] = raw["quarter"].astype(str)
        raw["Quarter"] = raw["year"] + " " + raw["quarter"]
        raw["FY_Start"] = raw["year"].str[:4].astype(int)
        raw["Q_Number"] = (
            raw["quarter"].str.extract(r"Q(\d+)")[0].astype(int)
        )

        duplicate_mask = raw.duplicated(
            subset=["base_year", "year", "quarter"], keep=False
        )
        if duplicate_mask.any():
            duplicates = raw.loc[
                duplicate_mask, ["year", "quarter"]
            ]
            raise ValueError(
                "Duplicate year-quarter observations found. Inspect the API "
                "revision information before plotting:\n"
                + duplicates.to_string(index=False)
            )

        raw = raw.sort_values(
            ["FY_Start", "Q_Number"]
        ).reset_index(drop=True)

        df = raw[
            ["Quarter", "FY_Start", "Q_Number",
             "Real_GDP_Growth", "Nominal_GDP_Growth"]
        ].copy()

        status_note = (
            f"Base year: {base_year} · "
            f"{len(df)} valid quarterly observations · "
            f"{total_pages} API page(s)"
        )
        return df, status_note

    except Exception as exc:
        raise RuntimeError(f"MoSPI GDP data retrieval failed: {exc}") from exc


# ---------------------------------------------------------------------
# CHART HELPERS
# ---------------------------------------------------------------------
PLOTLY_DARK = {
    "paper_bgcolor": "#101a2b",
    "plot_bgcolor": "#101a2b",
    "font": {"color": "#e8eef8", "family": "Inter, Arial, sans-serif"},
    "xaxis": {
        "title": "Financial quarter",
        "showgrid": False,
        "tickfont": {"size": 10, "color": "#aabbd2"},
        "linecolor": "#34445c",
        "zerolinecolor": "#34445c",
    },
    "yaxis": {
        "title": "Growth rate (%)",
        "showgrid": True,
        "gridcolor": "#26344a",
        "zeroline": True,
        "zerolinecolor": "#667991",
        "zerolinewidth": 1,
    },
    "legend": {
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "left",
        "x": 0,
        "font": {"color": "#dce7f5"},
    },
    "margin": {"l": 55, "r": 25, "t": 60, "b": 70},
    "hovermode": "x unified",
}


def build_gdp_chart(data: pd.DataFrame, metric_view: str):
    """Build the GDP growth chart for the selected metric view."""
    fig = go.Figure()

    if metric_view in ("Both", "Real GDP"):
        fig.add_trace(
            go.Scatter(
                x=data["Quarter"],
                y=data["Real_GDP_Growth"],
                mode="lines+markers",
                name="Real GDP · Constant prices",
                line={"color": "#64a8ff", "width": 3},
                marker={"size": 6, "color": "#64a8ff"},
                hovertemplate="%{x}<br><b>%{y:.1f}%</b><extra>Real GDP</extra>",
            )
        )

    if metric_view in ("Both", "Nominal GDP"):
        fig.add_trace(
            go.Scatter(
                x=data["Quarter"],
                y=data["Nominal_GDP_Growth"],
                mode="lines+markers",
                name="Nominal GDP · Current prices",
                line={"color": "#f0ad62", "width": 2.5, "dash": "dash"},
                marker={"size": 5, "color": "#f0ad62"},
                hovertemplate="%{x}<br><b>%{y:.1f}%</b><extra>Nominal GDP</extra>",
            )
        )

    fig.update_layout(**PLOTLY_DARK, height=470)
    # Show only Q1 labels, while retaining every quarterly data point.
    tick_positions = data.loc[
        data["Q_Number"] == 1, "Quarter"
    ].tolist()

    tick_labels = [
        f"{quarter.split(' ')[0]}<br>(Q1)"
        for quarter in tick_positions
    ]

    fig.update_xaxes(
        tickangle=-35,
        nticks=min(12, max(4, len(data) // 2)),
        tickmode="array",
        tickvals=tick_positions,
        ticktext=tick_labels,
        showspikes=True,
        spikemode="across",
        spikethickness=1,
        spikecolor="#FC0C8C",
        spikedash="dash",
        showgrid=False,
        automargin=True,
    )
    fig.update_yaxes(ticksuffix="%")
    fig.add_hline(
        y=0,
        line_width=1,
        line_dash="dot",
        line_color="#667991",
        opacity=0.8,
    )
    return fig


# ---------------------------------------------------------------------
# SIDEBAR: CONTROLS AND SOURCE INFORMATION
# ---------------------------------------------------------------------
st.sidebar.markdown("## 🇮🇳 India Economic Monitor")
st.sidebar.caption("Official macroeconomic indicators")

st.sidebar.markdown("### Dataset configuration")
selected_base_year = st.sidebar.selectbox(
    "GDP base-year series",
    options=["2011-12", "2022-23"],
    index=0,
    help="Select the base-year series published by MoSPI.",
)

metric_view = st.sidebar.segmented_control(
    "Chart metric",
    options=["Both", "Real GDP", "Nominal GDP"],
    default="Both",
    help="Choose which GDP growth series to display.",
)

if st.sidebar.button("↻ Refresh official data", use_container_width=True):
    fetch_live_mospi_data.clear()
    st.rerun()


st.sidebar.markdown("---")

# Reserve this position for the API status.
# The actual status will be inserted after the API fetch succeeds.
status_placeholder = st.sidebar.empty()

st.sidebar.markdown("---")
st.sidebar.markdown("### Official sources")
st.sidebar.markdown(
    "- [MoSPI GDP dashboard](https://www.mospi.gov.in/themes/product/6-gross-domestic-product)\n"
    "- [e-Sankhyiki NAS portal](https://esankhyiki.mospi.gov.in/macroindicators?product=nas)\n"
    "- [Open Government Data](https://data.gov.in)\n"
    "- [RBI DBIE](https://dbie.rbi.org.in)"
)
st.sidebar.caption(
    "GDP estimates may be revised by the official statistical authority. "
    "This app displays the values returned by the selected MoSPI API series."
)


# ---------------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------------
st.markdown('<div class="eyebrow">Official data · Quarterly national accounts</div>',
            unsafe_allow_html=True)
st.title("India Economic Monitor")
st.markdown(
    "Explore India's quarterly GDP growth using data retrieved from "
    "**MoSPI e-Sankhyiki**."
)


try:
    with st.spinner("Retrieving official GDP data from MoSPI…"):
        df_live, status_note = fetch_live_mospi_data(selected_base_year)

    # Fill the sidebar placeholder after a successful fetch.
    with status_placeholder.container():
        st.markdown("### Data status")
        st.success(f"MoSPI API connected | {status_note}")

except RuntimeError:
    # Detailed traceback goes to the server logs, not the public dashboard.
    import logging
    logging.exception("MoSPI GDP data retrieval failed")

    with status_placeholder.container():
        st.markdown("### Data status")
        st.error("MoSPI API unavailable")

    st.error("We couldn't retrieve GDP data from the official source.")
    st.info(
        "Please check your internet connection and try refreshing later. "
        "The MoSPI API may also be temporarily unavailable."
    )
    st.stop()

if df_live.empty:
    st.warning("No GDP data is available for the selected base-year series.")
    st.stop()

# This is the difference between the two growth rates, not a measured
# GDP-deflator inflation rate.
df_live["Growth_Gap_pp"] = (
    df_live["Nominal_GDP_Growth"] - df_live["Real_GDP_Growth"]
)

# Keep the data in chronological order and use the latest row for KPIs.
latest = df_live.iloc[-1]
previous = df_live.iloc[-2] if len(df_live) >= 2 else None
latest_quarter = str(latest["Quarter"])

# Calculate absolute changes in percentage points.
# For the growth gap, compare the gap against the previous observation.
if previous is not None:
    real_delta = (
        latest["Real_GDP_Growth"] - previous["Real_GDP_Growth"]
    )
    nominal_delta = (
        latest["Nominal_GDP_Growth"] - previous["Nominal_GDP_Growth"]
    )

    latest_gap = (
        latest["Nominal_GDP_Growth"] - latest["Real_GDP_Growth"]
    )
    previous_gap = (
        previous["Nominal_GDP_Growth"] - previous["Real_GDP_Growth"]
    )
    gap_delta = latest_gap - previous_gap
else:
    real_delta = nominal_delta = gap_delta = None

# Filter by financial year, without changing the underlying source data.
available_years = sorted(df_live["FY_Start"].unique().tolist())
if len(available_years) > 1:
    default_range = (available_years[0], available_years[-1])
    year_range = st.select_slider(
        "Financial-year range",
        options=available_years,
        value=default_range,
        format_func=lambda year: f"{year}-{str(year + 1)[-2:]}",
        help="Choose the first and last financial year to include in the chart.",
    )
    start_year, end_year = year_range
    chart_data = df_live[
        df_live["FY_Start"].between(start_year, end_year)
    ].copy()
else:
    chart_data = df_live.copy()
    st.caption(f"Financial year available: {available_years[0]}-{str(available_years[0] + 1)[-2:]}")

# Data status and latest observation.
calendar_col, latest_col = st.columns([2, 1], gap="medium")
with calendar_col:
    st.markdown("### Quarterly GDP release calendar")
    st.caption(
        "Typical publication schedule for quarterly GDP estimates. "
        "Actual release dates may vary."
    )

    release_cols = st.columns(4)

    release_schedule = [
        ("Q1", "April – June", "End of August"),
        ("Q2", "July – September", "End of November"),
        ("Q3", "October – December", "End of February"),
        ("Q4", "January – March", "End of May"),
    ]

    for col, (quarter, months, publication) in zip(
        release_cols, release_schedule
    ):
        with col:
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="eyebrow">{quarter}</div>
                    <div style="font-size:1.05rem;font-weight:600;
                                margin-bottom:.5rem">{months}</div>
                    <div class="muted" style="font-size:.85rem">
                        Published<br>
                        <strong>{publication}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


with latest_col:
    # Align the latest-quarter card with the centre of the calendar cards.
    st.markdown(
        '<div style="height: 6.2rem;"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="section-card"
             style="
                height: 140px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                justify-content: center;
                padding: 10px 10px;
             ">
            <div class="muted">Latest observation</div>
            <h2 style="margin: 5px 0 0 0;">
                {latest_quarter}
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### At a glance")
k1, k2, k3 = st.columns(3)
with k1:
    st.metric(
        label="Real GDP growth",
        value=f"{latest['Real_GDP_Growth']:.1f}%",
        delta=(
            f"{real_delta:+.1f} pp"
            if real_delta is not None else None
        ),
        delta_color="normal",
        help="Change from the previous available quarter, in percentage points.",
    )

with k2:
    st.metric(
        label="Nominal GDP growth",
        value=f"{latest['Nominal_GDP_Growth']:.1f}%",
        delta=(
            f"{nominal_delta:+.1f} pp"
            if nominal_delta is not None else None
        ),
        delta_color="normal",
        help="Change from the previous available quarter, in percentage points.",
    )

with k3:
    st.metric(
        label="Nominal–real growth gap",
        value=f"{latest_gap:.1f} pp",
        delta=(
            f"{gap_delta:+.1f} pp"
            if gap_delta is not None else None
        ),
        delta_color="normal",
        help="Change in the nominal–real growth gap from the previous available quarter.",
    )

st.markdown("---")
chart_title_col, chart_help_col = st.columns([3, 2])
with chart_title_col:
    st.markdown("### Quarterly GDP growth")
    st.caption(
        f"Base year {selected_base_year} · "
        f"{len(chart_data)} quarter(s) in selected range"
    )
with chart_help_col:
    st.caption(
        "Hover over a point for the quarter and value. Use the chart toolbar "
        "to zoom, pan, or save an image."
    )

if chart_data.empty:
    st.warning("No observations fall within the selected financial-year range.")
else:
    fig = build_gdp_chart(chart_data, metric_view or "Both")
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={"displaylogo": False, "responsive": True})

with st.expander("View quarterly data", expanded=False):
    display_df = df_live[
        ["Quarter", "Real_GDP_Growth", "Nominal_GDP_Growth", "Growth_Gap_pp"]
    ].copy()
    display_df = display_df.rename(
        columns={
            "Quarter": "Financial quarter",
            "Real_GDP_Growth": "Real GDP growth (%)",
            "Nominal_GDP_Growth": "Nominal GDP growth (%)",
            "Growth_Gap_pp": "Nominal - real growth gap (pp)",
        }
    )
    st.dataframe(
        display_df.style.format(
            {
                "Real GDP growth (%)": "{:.1f}",
                "Nominal GDP growth (%)": "{:.1f}",
                "Nominal - real growth gap (pp)": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    csv_bytes = display_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download quarterly data (CSV)",
        data=csv_bytes,
        file_name=f"india_gdp_growth_base_{selected_base_year}.csv",
        mime="text/csv",
    )

with st.expander("Data source and methodology"):
    st.markdown(
        f"""
        - **Provider:** Ministry of Statistics and Programme Implementation (MoSPI).
        - **Dataset:** National Accounts Statistics (NAS), GDP Growth Rate.
        - **Base-year series:** {selected_base_year}.
        - **Frequency:** Quarterly.
        - **Series selection:** Current.
        - **Displayed metrics:** Constant-price growth (real) and current-price
          growth (nominal).
        - **Data handling:** API pages are retrieved, reference-year records
          with both growth values missing are excluded, and observations are
          sorted chronologically. Records with only one growth value missing
          cause a visible data error rather than being silently plotted.
        - **Caching:** Successful results are cached for up to one hour; use
          “Refresh official data” in the sidebar to request a fresh fetch.
        - **Revisions:** Official GDP estimates can be revised. Values may
          differ across official releases or interfaces.
        """
    )

st.markdown("---")
st.caption(
    "India Economic Monitor · Source values are displayed from the selected "
    "MoSPI API series."
)
