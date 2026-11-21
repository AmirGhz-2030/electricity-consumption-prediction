import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
import joblib
import json
import os

st.set_page_config(page_title="Energy Consumption Predictor", page_icon="⚡", layout="wide")

# --- استایل‌های سازمانی مینی‌مال و فوق پایدار (الگوبرداری شده از داشبورد اول شما) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
    .stApp { background-color: #0b1120 !important; }
    html, body, [class*="css"], [class*="st-"] { font-family: 'Vazirmatn', sans-serif !important; direction: rtl; text-align: right; color: #e2e8f0 !important; }
    [data-testid="stSidebar"] { background-color: #111827 !important; border-left: 1px solid #1f2937 !important; }
    div[data-testid="metric-container"] { background-color: #1f2937 !important; border: 1px solid #374151 !important; padding: 15px !important; border-radius: 8px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important; }
    div[data-testid="stMetricValue"] { font-size: 22px !important; white-space: normal !important; overflow-wrap: break-word !important; }
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div { background-color: #111827 !important; color: #f8fafc !important; border: 1px solid #374151 !important; border-radius: 8px !important; }
    .stButton > button { background-color: #2563eb !important; color: white !important; border-radius: 8px !important; border: none !important; font-weight: bold !important; width: 100%; padding: 10px !important;}
    .stButton > button:hover { background-color: #1d4ed8 !important; }
    .stDownloadButton > button { background-color: #10b981 !important; color: white !important; width: 100%;}
    
    .deviation-grid { display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px; direction: rtl;}
    .deviation-card { background-color: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 12px 20px; min-width: 140px; text-align: center; transition: transform 0.2s;}
    .deviation-card:hover { transform: translateY(-3px); }
    .deviation-title { font-size: 14px; font-weight: bold; color: #94a3b8; margin-bottom: 5px;}
    .val-primary { color: #3b82f6; font-family: monospace; font-size: 16px; font-weight: bold; }
    .val-success { color: #10b981; font-family: monospace; font-size: 16px; font-weight: bold; }
    .val-warning { color: #f59e0b; font-family: monospace; font-size: 16px; font-weight: bold; }
    header { background-color: transparent !important; }
    
    div[data-baseweb="popover"] > div, div[data-testid="stPopoverBody"], div[role="dialog"] { background-color: #1f2937 !important; border: 1px solid #374151 !important; }
    ul[data-testid="main-menu-list"], ul[data-baseweb="menu"], ul[role="listbox"] { background-color: transparent !important; }
    ul[data-testid="main-menu-list"] li, ul[data-baseweb="menu"] li, ul[role="listbox"] li, li[role="option"], div[role="menuitem"] { background-color: #1f2937 !important; border: none !important; }
    div[data-baseweb="popover"] *, div[role="dialog"] *, div[data-baseweb="select"] * { color: #e2e8f0 !important; }
    ul[data-testid="main-menu-list"] li:hover, ul[data-baseweb="menu"] li:hover, ul[role="listbox"] li:hover, li[role="option"]:hover, div[role="menuitem"]:hover { background-color: #374151 !important; cursor: pointer !important; }
    code { background-color: #111827 !important; color: #93c5fd !important; border: 1px solid #374151 !important; padding: 2px 6px !important; border-radius: 4px !important; }
    
    [data-testid="stFileUploader"] section { background-color: #111827 !important; border: 2px dashed #3b82f6 !important; border-radius: 8px !important; padding: 20px !important; }
    [data-testid="stFileUploader"] section * { color: #cbd5e1 !important; }
    [data-testid="stFileUploader"] section button { background-color: #1f2937 !important; color: #60a5fa !important; border: 1px solid #374151 !important; font-weight: bold !important; }
    [data-testid="stDataFrame"] { border: 1px solid #374151 !important; border-radius: 8px !important; overflow: hidden !important; }
    </style>
""", unsafe_allow_html=True)

# --- خواندن پویای شناسنامه ارزیابی مدل‌ها (R² Scores اصلاح شده بر اساس ارائه رسمی) ---
metrics_file_path = "energy_metrics.json"
r2_linear = "محاسبه نشده"
r2_mlp = "محاسبه نشده"
r2_xgboost = "محاسبه نشده"

if os.path.exists(metrics_file_path):
    try:
        with open(metrics_file_path, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            r2_linear = meta_data.get('linear_r2', '0.7470')
            r2_mlp = meta_data.get('mlp_r2', '0.7430')
            r2_xgboost = meta_data.get('xgboost_r2', '0.6730')
    except Exception:
        r2_linear = r2_mlp = r2_xgboost = "خطا در فایل"
else:
    # مقادیر دقیق استخراج شده مستقیم از مستندات ارائه شما
    r2_linear, r2_mlp, r2_xgboost = "0.7470", "0.7430", "0.6730"

# --- بارگذاری هوشمند فایل‌های وزنی مدل‌ها و اسکیلر ---
@st.cache_resource
def load_energy_models():
    resources = {}
    if not os.path.exists('scaler.joblib'):
        raise FileNotFoundError("اسکیلر پیدا نشد.")
        
    resources['scaler'] = joblib.load('scaler.joblib')
    
    if os.path.exists('linear_model.joblib'):
        resources['linear'] = joblib.load('linear_model.joblib')
    if os.path.exists('mlp_model.joblib'):
        resources['mlp'] = joblib.load('mlp_model.joblib')
    if os.path.exists('xgboost_model.joblib'):
        resources['xgb'] = joblib.load('xgboost_model.joblib')
        
    return resources

try:
    assets = load_energy_models()
    scaler = assets['scaler']
except Exception:
    st.error("خطا: فایل مدل‌ها یا اسکیلر (scaler.joblib) یافت نشد. لطفاً ابتدا نوت‌بوک‌ها را تا سلول آخر ران کنید تا فایل‌ها استخراج شوند.")
    st.stop()

# --- اسامی دقیق ویژگی‌ها متناسب با ترتیب آموزش شما ---
feature_columns = ['house_size_m2', 'num_residents', 'avg_daily_temp', 'heating_cooling', 'high_power_appliances']

# --- سایدبار مشخصات سیستم ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2731/2731636.png", width=90)
    st.markdown("### ⚙️ مشخصات معماری سیستم")
    st.info("نوع مدل‌ها:\n\n**Linear Baseline / MLP Regressor / XGBoost**")
    st.success("وضعیت پردازش: **پایدار و بهینه‌سازی شده ✅**")
    st.divider()
    st.caption(" داشبورد هوشمند")
    st.markdown("<h4 style='color: #60a5fa;'>👨‍💻 سازنده: AmirGhz-2030</h4>", unsafe_allow_html=True)

# --- بدنه اصلی داشبورد ---
st.markdown("<h2 style='color: #60a5fa;'>داشبورد هوشمند پیش‌بینی و تحلیل مصرف برق (Energy Regression)</h2>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8;'>مقایسه تطبیقی سیستم رگرسیون خطی، شبکه عصبی پیشخور عمیق (MLPRegressor) و فریم‌ورک درختی XGBoost همراه با شاخص دقت مدل‌ها.</p>", unsafe_allow_html=True)

# ردیف کارت‌های متحرک متریک سیستم - نمایش زنده ضریب تبیین (R² Score) مدل‌ها بر اساس ارائه رسمی شما
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("دقت رگرسیون خطی", f"R²: {r2_linear}", "مدل برتر (Best Baseline)") 
col_m2.metric("دقت شبکه عصبی MLP", f"R²: {r2_mlp}", "معماری (128, 64)")
col_m3.metric("دقت مدل XGBoost", f"R²: {r2_xgboost}", "الگوریتم درختی تجمعی")
col_m4.metric("وضعیت نهایی تانسورها", "StandardScaler", "خوانده شده از خط لوله 📂")
st.divider()

tab1, tab2 = st.tabs(["🎯 پیش‌بینی انفرادی بار (ورودی دستی سناریو)", "📊 پردازش گروهی و کلان (آپلود فایل CSV تست)"])

with tab1:
    st.markdown("#### 🎛️ فاکتورهای محیطی و ساختاری نمونه واحد:")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        house_size = st.number_input("📐 متراژ بنا (مترمربع):", min_value=10.0, max_value=500.0, value=100.0)
        num_residents = st.number_input("👥 تعداد ساکنین واحد ساختمانی:", min_value=1, max_value=20, value=3)
        avg_temp = st.number_input("🌡️ میانگین دمای روزانه هوا (سانتی‌گراد):", min_value=-20.0, max_value=60.0, value=25.0)
    with col_in2:
        heating_cooling = st.selectbox("❄️🔥 وضعیت سیستم سرمایش یا گرمایش برقی:", ["خاموش (0)", "روشن (1)"])
        high_appliances = st.number_input("🔌 تعداد وسایل پرمصرف همزمان روشن:", min_value=0, max_value=20, value=2)

    # همگام‌سازی انتخاب متنی با دیتابیس عددی
    hc_val = 1 if "روشن" in heating_cooling else 0

    if st.button("شروع استنتاج و تخمین مصرف انرژی"):
        with st.spinner("در حال فراخوانی وزن‌ها و پردازش تانسورهای ورودی..."):
            # ساخت دیتافریم خام با ترتیب دقیق ویژگی‌ها
            input_data = pd.DataFrame([{
                'house_size_m2': house_size,
                'num_residents': num_residents,
                'avg_daily_temp': avg_temp,
                'heating_cooling': hc_val,
                'high_power_appliances': high_appliances
            }])[feature_columns]
            
            # اعمال StandardScaler روی ورودی جدید کاربر
            scaled_input = scaler.transform(input_data)
            
            # دیکشنری برای جمع‌آوری خروجی پیش‌بینی‌ها
            preds = {}
            if 'linear' in assets: preds['Linear Regression'] = assets['linear'].predict(scaled_input)[0]
            if 'mlp' in assets: preds['MLP Neural Network'] = assets['mlp'].predict(scaled_input)[0]
            if 'xgb' in assets: preds['XGBoost Regressor'] = assets['xgb'].predict(scaled_input)[0]
            
            st.markdown("<br>#### 🎯 تخمین نهایی مدل‌های رگرسیون بر حسب کیلووات ساعت (kWh):", unsafe_allow_html=True)
            
            # نمایش خروجی‌ها در قالب کارت‌های تفکیک شده
            html_cards = '<div class="deviation-grid">'
            if 'Linear Regression' in preds:
                html_cards += f'<div class="deviation-card"><div class="deviation-title">Linear Regression</div><div class="val-primary">{preds["Linear Regression"]:.2f} kWh</div></div>'
            if 'MLP Neural Network' in preds:
                html_cards += f'<div class="deviation-card"><div class="deviation-title">MLP Neural Network</div><div class="val-success">{preds["MLP Neural Network"]:.2f} kWh</div></div>'
            if 'XGBoost Regressor' in preds:
                html_cards += f'<div class="deviation-card"><div class="deviation-title">XGBoost Regressor</div><div class="val-warning">{preds["XGBoost Regressor"]:.2f} kWh</div></div>'
            html_cards += '</div>'
            st.markdown(html_cards, unsafe_allow_html=True)
            
            # رسم نمودار مقایسه‌ای ستونی تعاملی با پلاتلی
            fig_user = px.bar(
                x=list(preds.keys()), y=list(preds.values()), 
                labels={'x': 'معماری مدل‌های ارزیابی شده', 'y': 'مصرف انرژی تخمینی (kWh)'},
                title="📊 انحراف پیش‌بینی مدل‌ها نسبت به یکدیگر", template="plotly_dark",
                color=list(preds.keys()), color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b']
            )
            fig_user.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_user, use_container_width=True)

with tab2:
    st.markdown("#### 📊 پردازش کلان داده‌های مصرف انرژی (Batch Inference)")
    uploaded_file = st.file_uploader("فایل CSV تست (مثلاً test.csv) را انتخاب کنید", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        # بررسی انطباق ستون‌های فایل ورودی با ماتریس ویژگی‌ها
        missing_cols = [col for col in feature_columns if col not in batch_df.columns]
        
        if missing_cols:
            st.error(f"⚠️ ساختار ستون‌های فایل آپلود شده با ماتریس ویژگی‌های مدل همخوانی ندارد. ستون‌های گمشده: {missing_cols}")
        else:
            st.success(f"فایل با موفقیت فراخوانی شد. تعداد فاکتورهای شناسایی شده: {len(batch_df)} رکورد بار الکتریکی.")
            
            if st.button("🚀 آغاز فرآیند استنتاج همزمان کلان داده"):
                with st.spinner("در حال انتقال بردارهای ویژگی به اسکیلر و اجرای موازی تخمین‌ها..."):
                    # جداسازی ویژگی‌های مورد نیاز و مقیاس‌بندی آن‌ها
                    X_batch_raw = batch_df[feature_columns]
                    X_batch_scaled = scaler.transform(X_batch_raw)
                    
                    # ثبت ستون‌های تخمینی جدید برای هر مدل روی دیتافریم اصلی
                    if 'linear' in assets: batch_df['Pred_Linear_Regression'] = assets['linear'].predict(X_batch_scaled)
                    if 'mlp' in assets: batch_df['Pred_MLP_Network'] = assets['mlp'].predict(X_batch_scaled)
                    if 'xgb' in assets: batch_df['Pred_XGBoost'] = assets['xgb'].predict(X_batch_scaled)
                    
                    rep_col1, rep_col2 = st.columns([1, 2])
                    
                    with rep_col1:
                        st.markdown("##### 📈 خلاصه آمار محاسباتی:")
                        st.write(f"🔹 حجم داده پردازش شده: **{len(batch_df):,} رکورد**")
                        
                        # محاسبه میانگین مصرف تخمینی هر مدل برای گزارش کلان
                        if 'Pred_Linear_Regression' in batch_df.columns:
                            st.write(f"🔵 میانگین مصرف رگرسیون: **{batch_df['Pred_Linear_Regression'].mean():.2f} kWh**")
                        if 'Pred_MLP_Network' in batch_df.columns:
                            st.write(f"🟢 میانگین مصرف شبکه عصبی: **{batch_df['Pred_MLP_Network'].mean():.2f} kWh**")
                        if 'Pred_XGBoost' in batch_df.columns:
                            st.write(f"🟡 میانگین مصرف ایکس‌جی‌بوست: **{batch_df['Pred_XGBoost'].mean():.2f} kWh**")
                        
                        # ساخت لایه دانلود مستقیم اکسل خروجی نتایج
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            batch_df.to_excel(writer, index=False)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.download_button("📥 دانلود نتایج جامع مقایسه‌ای (Excel)", data=buffer.getvalue(), file_name="Energy_Predictions_Comparative_Results.xlsx", mime="application/vnd.ms-excel", use_container_width=True)
                    
                    with rep_col2:
                        st.markdown("##### 🎛️ تحلیل فرکانسی و نوسان رفتار مدل‌ها (۱۰۰ نمونه اول):")
                        pred_cols = [c for c in batch_df.columns if 'Pred_' in c]
                        plot_df = batch_df.head(100).reset_index()
                        
                        fig_line = px.line(
                            plot_df, x='index', y=pred_cols, 
                            labels={'value': 'سطح انرژی مصرفی (kWh)', 'index': 'شناسه ردیف نمونه'},
                            template="plotly_dark"
                        )
                        fig_line.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_line, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("##### 🔎 پیش‌نمایش تلاقی پیش‌بینی مدل‌ها (۱۰ رکورد ابتدایی):")
                    show_cols = feature_columns + pred_cols
                    st.dataframe(batch_df[show_cols].head(10), use_container_width=True)
