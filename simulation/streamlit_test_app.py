import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Sentinel360 Model Tester",
    page_icon="\U0001f697",
    layout="wide"
)

st.title("\U0001f697 Sentinel360 Hybrid Model Testing Suite")
st.markdown("**Test edge cases and analyze model predictions vs expected outcomes**")


@st.cache_resource
def load_models():
    try:
        ghana_gb = joblib.load('models/ghana_gb_model.pkl')
        porto_if = joblib.load('models/porto_if_model.pkl')
        scaler_gh = joblib.load('models/ghana_scaler.pkl')
        scaler_po = joblib.load('models/porto_scaler.pkl')
        features = joblib.load('models/feature_names.pkl')
        config = joblib.load('models/fusion_config.pkl')
        return {
            'ghana_gb': ghana_gb,
            'porto_if': porto_if,
            'scaler_gh': scaler_gh,
            'scaler_po': scaler_po,
            'features': features,
            'config': config,
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


models = load_models()
if models is None:
    st.stop()

FEATURE_NAMES = models['features']
WEIGHT_GB = models['config'].get('weight_gb', 0.7)
WEIGHT_IF = models['config'].get('weight_if', 0.3)
THRESHOLD_SAFE = models['config'].get('threshold_safe', 0.3)
THRESHOLD_HIGH = models['config'].get('threshold_high', 0.7)


def predict_hybrid(features_dict, models):
    feature_array = np.array([[features_dict.get(f, 0) for f in FEATURE_NAMES]])

    ghana_scaled = models['scaler_gh'].transform(feature_array)
    gb_score = float(models['ghana_gb'].predict_proba(ghana_scaled)[0][1])

    porto_scaled = models['scaler_po'].transform(feature_array)
    if_raw = float(models['porto_if'].decision_function(porto_scaled)[0])
    if_score = 1 / (1 + np.exp(-if_raw))

    hybrid_score = WEIGHT_GB * gb_score + WEIGHT_IF * if_score

    if hybrid_score < THRESHOLD_SAFE:
        level, color = 'SAFE', 'green'
    elif hybrid_score < THRESHOLD_HIGH:
        level, color = 'MEDIUM', 'orange'
    else:
        level, color = 'HIGH RISK', 'red'

    return {
        'gb_score': gb_score,
        'if_score': if_score,
        'hybrid_score': hybrid_score,
        'level': level,
        'color': color,
    }


# ── Test Scenarios (mapped to actual model features) ─────────────────────────
# Model features: speed, acceleration, acceleration_variation, trip_duration,
# trip_distance, stop_events, road_encoded, weather_encoded, traffic_encoded,
# hour, month, avg_speed, stops_per_km, accel_abs, speed_normalized,
# speed_squared, is_rush_hour, is_night

def _scenario(speed, accel, accel_var, duration, distance, stops,
              road=0, weather=0, traffic=1, hour=10, month=2,
              rush=0, night=0):
    avg_spd = distance / (duration / 3600 + 0.001)
    return {
        'speed': speed, 'acceleration': accel,
        'acceleration_variation': accel_var, 'trip_duration': duration,
        'trip_distance': distance, 'stop_events': stops,
        'road_encoded': road, 'weather_encoded': weather,
        'traffic_encoded': traffic, 'hour': hour, 'month': month,
        'avg_speed': avg_spd, 'stops_per_km': stops / (distance + 0.1),
        'accel_abs': abs(accel), 'speed_normalized': speed / 100,
        'speed_squared': speed ** 2, 'is_rush_hour': rush, 'is_night': night,
    }


TEST_SCENARIOS = {
    "1. Normal City Commute (Expected: SAFE)": {
        'description': "Regular driving at 40 km/h, gentle braking, few stops",
        'expected': 'SAFE',
        'features': _scenario(speed=40, accel=0.5, accel_var=1.2,
                              duration=900, distance=8.5, stops=3, hour=8, rush=1),
    },
    "2. Highway Cruise (Expected: SAFE)": {
        'description': "Steady 90 km/h, almost no acceleration change",
        'expected': 'SAFE',
        'features': _scenario(speed=90, accel=0.2, accel_var=0.8,
                              duration=1200, distance=25.0, stops=0, hour=14),
    },
    "3. Excessive Speeding (Expected: HIGH RISK)": {
        'description': "110 km/h in urban zone, hard acceleration, late night",
        'expected': 'HIGH RISK',
        'features': _scenario(speed=110, accel=4.5, accel_var=8.0,
                              duration=600, distance=15.0, stops=1, hour=23, night=1),
    },
    "4. Harsh Braking Pattern (Expected: HIGH RISK)": {
        'description': "Multiple emergency stops, high deceleration variance",
        'expected': 'HIGH RISK',
        'features': _scenario(speed=55, accel=-4.0, accel_var=9.5,
                              duration=900, distance=10.0, stops=8, hour=19, rush=1),
    },
    "5. Erratic City Driving (Expected: MEDIUM)": {
        'description': "Frequent speed changes, moderate acceleration bursts",
        'expected': 'MEDIUM',
        'features': _scenario(speed=48, accel=2.5, accel_var=5.5,
                              duration=1200, distance=12.0, stops=6, hour=17, rush=1),
    },
    "6. Professional Taxi Driver (Expected: SAFE)": {
        'description': "Smooth, experienced driving with passenger stops",
        'expected': 'SAFE',
        'features': _scenario(speed=42, accel=0.8, accel_var=1.5,
                              duration=960, distance=10.5, stops=4, hour=10),
    },
    "7. Edge: Single Emergency Brake (Expected: SAFE)": {
        'description': "Otherwise safe trip with one hard braking event",
        'expected': 'SAFE',
        'features': _scenario(speed=45, accel=-2.0, accel_var=3.0,
                              duration=840, distance=9.0, stops=2, hour=12),
    },
    "8. Slow but Suspicious (Expected: MEDIUM)": {
        'description': "Low speed, many stops, late night, meandering",
        'expected': 'MEDIUM',
        'features': _scenario(speed=25, accel=0.4, accel_var=2.0,
                              duration=1500, distance=5.0, stops=12, hour=2, night=1),
    },
    "9. Speed Limit Boundary (Expected: SAFE)": {
        'description': "Exactly at urban speed limit, near-zero variation",
        'expected': 'SAFE',
        'features': _scenario(speed=50, accel=0.2, accel_var=0.5,
                              duration=900, distance=12.0, stops=1, hour=15),
    },
    "10. Reckless Driving (Expected: HIGH RISK)": {
        'description': "Max speed + harsh accel + harsh braking + night",
        'expected': 'HIGH RISK',
        'features': _scenario(speed=130, accel=6.0, accel_var=12.0,
                              duration=720, distance=18.0, stops=2, hour=1, night=1),
    },
    "11. Perfect Cruise Control (Expected: SAFE)": {
        'description': "Constant 80 km/h, zero acceleration change, highway",
        'expected': 'SAFE',
        'features': _scenario(speed=80, accel=0.0, accel_var=0.2,
                              duration=1350, distance=30.0, stops=0, hour=11),
    },
    "12. Stop-and-Go Traffic Jam (Expected: SAFE)": {
        'description': "Very slow but normal traffic congestion behavior",
        'expected': 'SAFE',
        'features': _scenario(speed=18, accel=0.6, accel_var=1.8,
                              duration=1800, distance=6.0, stops=15, hour=8, rush=1),
    },
    "13. Trotro Aggressive Overtake (Expected: MEDIUM)": {
        'description': "Typical Accra minibus: sudden lane changes, honking speed",
        'expected': 'MEDIUM',
        'features': _scenario(speed=65, accel=3.0, accel_var=6.0,
                              duration=600, distance=8.0, stops=5, hour=7, rush=1),
    },
    "14. Wet-Road Night Drive (Expected: MEDIUM)": {
        'description': "Moderate speed on wet roads at night, occasional slides",
        'expected': 'MEDIUM',
        'features': _scenario(speed=55, accel=1.5, accel_var=4.5,
                              duration=1200, distance=14.0, stops=3, hour=22,
                              weather=1, night=1),
    },
    "15. Rush-Hour Motorway Merge (Expected: SAFE)": {
        'description': "Motorway entry ramp during peak, brief acceleration burst",
        'expected': 'SAFE',
        'features': _scenario(speed=70, accel=1.8, accel_var=2.5,
                              duration=300, distance=6.0, stops=1, hour=8, rush=1),
    },
}


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("Testing Options")
mode = st.sidebar.radio(
    "Select Mode",
    ["\U0001f4cb Pre-defined Scenarios", "\u270d\ufe0f Manual Input", "\U0001f4ca Batch Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.info(f"""
**Features:** {len(FEATURE_NAMES)}
**Ghana GB:** {type(models['ghana_gb']).__name__}
**Porto IF:** {type(models['porto_if']).__name__}
**Fusion:** {WEIGHT_GB}\u00d7GB + {WEIGHT_IF}\u00d7IF
**Safe threshold:** < {THRESHOLD_SAFE}
**High threshold:** \u2265 {THRESHOLD_HIGH}
""")


# ── Pre-defined Scenarios ────────────────────────────────────────────────────

if mode == "\U0001f4cb Pre-defined Scenarios":
    st.header("Pre-defined Test Scenarios")

    selected = st.selectbox("Choose a scenario", list(TEST_SCENARIOS.keys()))
    scenario = TEST_SCENARIOS[selected]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Scenario Details")
        st.write(f"**Description:** {scenario['description']}")
        exp = scenario['expected']
        st.write(f"**Expected Outcome:** **{exp}**")

        with st.expander("View Feature Values"):
            df_f = pd.DataFrame([scenario['features']]).T
            df_f.columns = ['Value']
            df_f.index.name = 'Feature'
            st.dataframe(df_f, use_container_width=True)

    with col2:
        st.subheader("Model Prediction")
        result = predict_hybrid(scenario['features'], models)

        st.metric("Hybrid Risk Score", f"{result['hybrid_score']:.3f}",
                  delta=result['level'])

        if result['color'] == 'green':
            st.success(f"\u2705 Classification: **{result['level']}**")
        elif result['color'] == 'orange':
            st.warning(f"\u26a0\ufe0f Classification: **{result['level']}**")
        else:
            st.error(f"\U0001f6a8 Classification: **{result['level']}**")

        col_gb, col_if = st.columns(2)
        with col_gb:
            st.metric("Ghana GB", f"{result['gb_score']:.3f}")
        with col_if:
            st.metric("Porto IF", f"{result['if_score']:.3f}")

        st.markdown("---")
        expected_lower = scenario['expected'].lower()
        actual_lower = result['level'].lower()
        if 'medium' in expected_lower:
            is_correct = 'medium' in actual_lower
        else:
            is_correct = expected_lower == actual_lower or expected_lower in actual_lower

        if is_correct:
            st.success("\u2705 **PASS** \u2014 Prediction matches expected outcome!")
        else:
            st.error(f"\u274c **FAIL** \u2014 Expected {exp}, got {result['level']}")

    st.subheader("Score Breakdown")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=result['hybrid_score'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Hybrid Risk Score"},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': result['color']},
            'steps': [
                {'range': [0, THRESHOLD_SAFE], 'color': "lightgreen"},
                {'range': [THRESHOLD_SAFE, THRESHOLD_HIGH], 'color': "lightyellow"},
                {'range': [THRESHOLD_HIGH, 1], 'color': "lightcoral"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': THRESHOLD_HIGH,
            },
        },
    ))
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=[f'Ghana GB ({WEIGHT_GB*100:.0f}%)',
           f'Porto IF ({WEIGHT_IF*100:.0f}%)', 'Hybrid'],
        y=[result['gb_score'], result['if_score'], result['hybrid_score']],
        marker_color=['#2563EB', '#10B981', result['color']],
    ))
    fig2.update_layout(title="Model Component Comparison",
                       yaxis_title="Risk Score", yaxis_range=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)


# ── Manual Input ─────────────────────────────────────────────────────────────

elif mode == "\u270d\ufe0f Manual Input":
    st.header("Manual Feature Input")
    st.write("Enter custom feature values to test specific driving scenarios")

    with st.form("manual_input"):
        col1, col2, col3 = st.columns(3)

        fi = {}
        with col1:
            st.subheader("Speed & Motion")
            fi['speed'] = st.number_input("Current Speed (km/h)", 0.0, 200.0, 45.0)
            fi['acceleration'] = st.number_input("Acceleration (m/s\u00b2)", -10.0, 10.0, 0.5)
            fi['acceleration_variation'] = st.number_input("Accel Variation (std)", 0.0, 20.0, 1.5)
            fi['accel_abs'] = st.number_input("Abs Acceleration", 0.0, 10.0, abs(fi['acceleration']))
            fi['speed_normalized'] = fi['speed'] / 100
            fi['speed_squared'] = fi['speed'] ** 2

        with col2:
            st.subheader("Trip Metrics")
            fi['trip_duration'] = st.number_input("Trip Duration (s)", 0.0, 7200.0, 900.0)
            fi['trip_distance'] = st.number_input("Trip Distance (km)", 0.0, 100.0, 10.0)
            fi['stop_events'] = st.number_input("Stop Events", 0, 50, 3)
            fi['avg_speed'] = fi['trip_distance'] / (fi['trip_duration'] / 3600 + 0.001)
            fi['stops_per_km'] = fi['stop_events'] / (fi['trip_distance'] + 0.1)
            st.caption(f"Computed avg speed: {fi['avg_speed']:.1f} km/h")
            st.caption(f"Computed stops/km: {fi['stops_per_km']:.2f}")

        with col3:
            st.subheader("Context")
            fi['hour'] = st.number_input("Hour (0-23)", 0, 23, 10)
            fi['month'] = st.number_input("Month (1-12)", 1, 12, 2)
            fi['road_encoded'] = st.selectbox("Road Type", [0, 1, 2],
                                              format_func=lambda x: ['Urban', 'Highway', 'Rural'][x])
            fi['weather_encoded'] = st.selectbox("Weather", [0, 1],
                                                 format_func=lambda x: ['Clear', 'Rain'][x])
            fi['traffic_encoded'] = st.selectbox("Traffic", [0, 1, 2],
                                                 format_func=lambda x: ['Light', 'Normal', 'Heavy'][x])
            fi['is_rush_hour'] = 1 if fi['hour'] in range(7, 10) or fi['hour'] in range(16, 19) else 0
            fi['is_night'] = 1 if fi['hour'] >= 22 or fi['hour'] <= 5 else 0

        submitted = st.form_submit_button("\U0001f52e Predict Risk")

    if submitted:
        result = predict_hybrid(fi, models)

        st.markdown("---")
        st.subheader("Prediction Result")

        cr1, cr2, cr3 = st.columns(3)
        with cr1:
            st.metric("Hybrid Score", f"{result['hybrid_score']:.3f}")
        with cr2:
            st.metric("Ghana GB", f"{result['gb_score']:.3f}")
        with cr3:
            st.metric("Porto IF", f"{result['if_score']:.3f}")

        if result['color'] == 'green':
            st.success(f"\u2705 **{result['level']}**")
        elif result['color'] == 'orange':
            st.warning(f"\u26a0\ufe0f **{result['level']}**")
        else:
            st.error(f"\U0001f6a8 **{result['level']}**")


# ── Batch Analysis ───────────────────────────────────────────────────────────

else:
    st.header("Batch Scenario Analysis")
    st.write("Running all pre-defined scenarios and analyzing results\u2026")

    if st.button("\U0001f680 Run All Tests"):
        results = []
        progress = st.progress(0)

        for idx, (name, scenario) in enumerate(TEST_SCENARIOS.items()):
            result = predict_hybrid(scenario['features'], models)

            expected_lower = scenario['expected'].lower()
            actual_lower = result['level'].lower()
            if 'medium' in expected_lower:
                is_correct = 'medium' in actual_lower
            else:
                is_correct = expected_lower == actual_lower or expected_lower in actual_lower

            results.append({
                'Scenario': name,
                'Expected': scenario['expected'],
                'Predicted': result['level'],
                'Hybrid Score': round(result['hybrid_score'], 3),
                'GB Score': round(result['gb_score'], 3),
                'IF Score': round(result['if_score'], 3),
                'Correct': '\u2705' if is_correct else '\u274c',
                'Match': is_correct,
            })
            progress.progress((idx + 1) / len(TEST_SCENARIOS))

        df = pd.DataFrame(results)

        st.subheader("Summary Statistics")
        c1, c2, c3, c4 = st.columns(4)
        accuracy = df['Match'].sum() / len(df) * 100
        with c1:
            st.metric("Overall Accuracy", f"{accuracy:.1f}%")
        with c2:
            st.metric("Total Tests", len(df))
        with c3:
            st.metric("Passed", int(df['Match'].sum()))
        with c4:
            st.metric("Failed", int((~df['Match']).sum()))

        st.subheader("Detailed Results")
        display_cols = ['Scenario', 'Expected', 'Predicted', 'Hybrid Score',
                        'GB Score', 'IF Score', 'Correct']
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        st.subheader("Prediction Distribution")
        confusion = pd.crosstab(df['Expected'], df['Predicted'], margins=True)
        st.dataframe(confusion)

        st.subheader("Score Distribution by Expected Outcome")
        fig = px.box(df, x='Expected', y='Hybrid Score', color='Expected',
                     points='all', title='Hybrid Scores by Expected Classification')
        fig.add_hline(y=THRESHOLD_SAFE, line_dash="dash", line_color="green",
                      annotation_text="Safe Threshold")
        fig.add_hline(y=THRESHOLD_HIGH, line_dash="dash", line_color="red",
                      annotation_text="High Risk Threshold")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Ghana GB vs Porto IF Scores")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df['GB Score'], y=df['IF Score'],
            mode='markers+text',
            marker=dict(size=15, color=df['Hybrid Score'],
                        colorscale='RdYlGn_r', showscale=True,
                        colorbar=dict(title="Hybrid")),
            text=[s.split('.')[0] for s in df['Scenario']],
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>GB: %{x:.3f}<br>IF: %{y:.3f}<extra></extra>',
        ))
        fig2.update_layout(xaxis_title="Ghana GB Score", yaxis_title="Porto IF Score",
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            label="\U0001f4e5 Download Results CSV",
            data=df.to_csv(index=False),
            file_name=f"sentinel360_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

st.markdown("---")
st.markdown("**Sentinel360 Model Testing Suite** | Ghana GB + Porto IF Hybrid Fusion")
