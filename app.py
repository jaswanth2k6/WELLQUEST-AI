import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# ============================================================
# WELLQUEST AI | Student wellbeing assessment prototype
# ============================================================

st.set_page_config(
    page_title="WELLQUEST AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.joblib"

FEATURES = [
    "anxiety_level", "self_esteem", "mental_health_history", "depression",
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns", "social_support", "peer_pressure",
    "extracurricular_activities", "bullying",
]

# Observed min/max values in the selected research dataset.
# These are dataset-coded values, not validated questionnaire anchors.
RANGES = {
    "anxiety_level": (0, 21),
    "self_esteem": (0, 30),
    "mental_health_history": (0, 1),
    "depression": (0, 27),
    "headache": (0, 5),
    "blood_pressure": (1, 3),
    "sleep_quality": (0, 5),
    "breathing_problem": (0, 5),
    "noise_level": (0, 5),
    "living_conditions": (0, 5),
    "safety": (0, 5),
    "basic_needs": (0, 5),
    "academic_performance": (0, 5),
    "study_load": (0, 5),
    "teacher_student_relationship": (0, 5),
    "future_career_concerns": (0, 5),
    "social_support": (0, 3),
    "peer_pressure": (0, 5),
    "extracurricular_activities": (0, 5),
    "bullying": (0, 5),
}


# ----------------------------- AI / VISUAL HELPERS -----------------------------
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,14,24,.72)",
    font=dict(color="#DCEAF5", family="DM Sans"),
    margin=dict(l=18, r=18, t=38, b=18),
    xaxis=dict(gridcolor="rgba(130,170,200,.13)", zerolinecolor="rgba(130,170,200,.16)"),
    yaxis=dict(gridcolor="rgba(130,170,200,.13)", zerolinecolor="rgba(130,170,200,.16)"),
)
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def make_donut(labels, values, title="Model estimates"):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=.70, sort=False,
        marker=dict(colors=["#13C8E8", "#8B5CF6", "#FF4FA3", "#24D18B"],
                    line=dict(color="#0B111B", width=3)),
        textinfo="percent", textfont=dict(size=13, color="white"),
        hovertemplate="%{label}<br>%{value:.1f}%<extra></extra>"
    ))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, x=.04, font=dict(size=15)),
                      showlegend=True, legend=dict(orientation="h", y=-.08, x=.05),
                      height=300)
    return fig


def make_radar(values, title="Indicator profile"):
    labels = [k.replace("_", " ").title() for k in values]
    vals = list(values.values())
    if labels:
        labels += [labels[0]]
        vals += [vals[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=labels, fill="toself", name="Range position",
        line=dict(color="#20C9EA", width=2),
        fillcolor="rgba(32,201,234,.20)",
        marker=dict(color="#FF4FA3", size=5)
    ))
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, x=.04, font=dict(size=15)),
        polar=dict(bgcolor="rgba(8,14,24,.55)", radialaxis=dict(visible=True, range=[0,100],
        gridcolor="rgba(130,170,200,.15)", tickfont=dict(color="#93A7B8", size=9)),
        angularaxis=dict(gridcolor="rgba(130,170,200,.15)", tickfont=dict(color="#C9D7E4", size=9))),
        showlegend=False, height=340)
    return fig


def range_positions(values, features):
    out = {}
    for feat in features:
        lo, hi = RANGES[feat]
        out[feat] = round((float(values.get(feat, lo)) - lo) / (hi - lo) * 100, 1) if hi > lo else 0
    return out


def model_probability_frame(model, frame):
    if model is None or not hasattr(model, "predict_proba"):
        return None
    probs = model.predict_proba(frame)[0]
    classes = getattr(model, "classes_", list(range(len(probs))))
    return pd.DataFrame({"Dataset class": [f"Class {c}" for c in classes], "Estimate (%)": [float(x)*100 for x in probs]})

# ----------------------------- DESIGN -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--black:#0A0A0A;--blue:#007BA7;--rose:#FF2D95;--white:#F5F7FA;--muted:#A3AAB7}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
html,body{background:#0A0A0A}
.stApp{
  position:relative;isolation:isolate;overflow-x:hidden;
  background:#080B11;color:var(--white)
}
/* Full-page ambient aurora: slow, blurred light fields behind all content. */
.stApp::before{
  content:"";position:fixed;z-index:0;pointer-events:none;
  inset:-28%;
  background:
    radial-gradient(ellipse at 12% 22%,rgba(0,190,235,.34) 0%,transparent 30%),
    radial-gradient(ellipse at 78% 18%,rgba(0,123,167,.30) 0%,transparent 31%),
    radial-gradient(ellipse at 82% 78%,rgba(255,45,149,.23) 0%,transparent 32%),
    radial-gradient(ellipse at 28% 86%,rgba(56,90,190,.24) 0%,transparent 35%),
    radial-gradient(ellipse at 52% 48%,rgba(0,180,170,.10) 0%,transparent 42%),
    linear-gradient(135deg,#050A10 5%,#071722 48%,#100A18 100%);
  background-size:140% 140%;
  filter:blur(38px);transform:translate3d(0,0,0) scale(1.08);
  animation:auroraDrift 22s ease-in-out infinite alternate
}

/* Premium ambient glow ribbons and responsive motion */
[data-testid="stAppViewContainer"]::before{
  content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.55;
  background:
    radial-gradient(ellipse 34% 22% at 18% 35%,rgba(0,190,235,.13),transparent 72%),
    radial-gradient(ellipse 30% 26% at 84% 55%,rgba(255,45,149,.12),transparent 72%),
    radial-gradient(ellipse 24% 20% at 50% 92%,rgba(0,123,167,.15),transparent 72%);
  background-size:160% 160%;animation:glowWander 26s ease-in-out infinite alternate
}
[data-testid="stMain"] > div{position:relative;z-index:1}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > div{animation:contentReveal .65s ease both}
[data-testid="stMain"] h1{letter-spacing:-1.6px;text-shadow:0 0 28px rgba(0,190,235,.16)}
[data-testid="stMain"] [data-testid="stNumberInput"] input,
[data-testid="stMain"] [data-testid="stSelectbox"] div[data-baseweb="select"]{
  background:rgba(20,27,39,.82)!important;border:1px solid rgba(91,196,226,.20)!important;
  border-radius:12px!important;transition:border-color .25s,box-shadow .25s,transform .25s!important
}
[data-testid="stMain"] [data-testid="stNumberInput"]:focus-within input,
[data-testid="stMain"] [data-testid="stSelectbox"]:focus-within div[data-baseweb="select"]{
  border-color:rgba(0,190,235,.72)!important;box-shadow:0 0 0 3px rgba(0,190,235,.10),0 0 25px rgba(0,190,235,.12)!important
}
[data-testid="stMain"] [data-testid="stForm"] h3,
[data-testid="stMain"] [data-testid="stForm"] h4{color:#EAF8FF!important}
.assessment-hero{position:relative;overflow:hidden;padding:30px 34px;margin:10px 0 22px;
  border-radius:24px;border:1px solid rgba(67,204,240,.30);
  background:linear-gradient(115deg,rgba(5,25,38,.92),rgba(8,54,73,.72) 52%,rgba(48,12,43,.72));
  box-shadow:0 18px 65px rgba(0,0,0,.28),inset 0 1px rgba(255,255,255,.07)}
.assessment-hero:before{content:"";position:absolute;width:260px;height:260px;right:-70px;top:-150px;border-radius:50%;
  background:radial-gradient(circle,rgba(0,207,255,.35),rgba(0,207,255,0) 68%);animation:orbPulse 7s ease-in-out infinite}
.assessment-hero:after{content:"";position:absolute;width:210px;height:210px;right:18%;bottom:-160px;border-radius:50%;
  background:radial-gradient(circle,rgba(255,45,149,.25),rgba(255,45,149,0) 70%);animation:orbPulse 9s ease-in-out infinite reverse}
.assessment-hero-inner{position:relative;z-index:1}
.assessment-eyebrow{font-size:10px;letter-spacing:3px;font-weight:800;color:#72DFF8;text-transform:uppercase}
.assessment-heading{font:700 clamp(30px,4vw,46px)/1.08 'Space Grotesk';letter-spacing:-1.5px;color:#fff;margin:12px 0}
.assessment-sub{max-width:720px;color:#C4D3DE;font-size:14px;line-height:1.75}
.assessment-chip{display:inline-block;margin-top:17px;margin-right:8px;padding:7px 11px;border:1px solid rgba(117,218,245,.22);
  border-radius:30px;background:rgba(4,20,31,.48);color:#D7F5FF;font-size:11px;letter-spacing:.4px}
.stage-row{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:0 0 18px}
.stage{padding:12px 14px;border-radius:13px;background:rgba(12,22,33,.72);border:1px solid rgba(95,197,225,.18);color:#AAB9C8;font-size:12px}
.stage b{display:block;color:#EAF8FF;font-size:13px;margin-bottom:4px}
.stage.active{border-color:rgba(0,190,235,.55);background:linear-gradient(110deg,rgba(0,123,167,.23),rgba(255,45,149,.08));box-shadow:0 0 22px rgba(0,160,210,.10)}
@keyframes glowWander{0%{transform:translate3d(-2%,1%,0) scale(1)}100%{transform:translate3d(2%,-2%,0) scale(1.12)}}
@keyframes contentReveal{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes orbPulse{0%,100%{transform:scale(.88);opacity:.55}50%{transform:scale(1.16);opacity:1}}

/* Tiny low-contrast light specks create depth without competing with text. */
.stApp::after{
  content:"";position:fixed;z-index:0;pointer-events:none;inset:0;
  opacity:.34;
  background-image:
    radial-gradient(1px 1px at 12% 22%,rgba(180,235,255,.65) 98%,transparent),
    radial-gradient(1px 1px at 72% 18%,rgba(180,235,255,.48) 98%,transparent),
    radial-gradient(1px 1px at 44% 67%,rgba(255,190,225,.42) 98%,transparent),
    radial-gradient(1px 1px at 86% 78%,rgba(180,235,255,.52) 98%,transparent),
    radial-gradient(1px 1px at 25% 88%,rgba(180,235,255,.40) 98%,transparent);
  background-size:340px 300px,420px 360px,500px 420px,390px 330px,460px 390px;
  animation:softParticles 48s linear infinite
}
/* Keep Streamlit's actual interface above the decorative background layers. */
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="stMain"],
[data-testid="stBottom"],
footer{
  position:relative;
  z-index:1;
}
[data-testid="stHeader"]{background:rgba(10,10,10,.62);backdrop-filter:blur(14px)}
[data-testid="stSidebar"]{
  background:linear-gradient(165deg,rgba(14,20,29,.98),rgba(7,10,15,.99) 58%,rgba(24,8,22,.98));
  border-right:1px solid rgba(0,183,235,.25);
  box-shadow:14px 0 45px rgba(0,0,0,.28);
}
[data-testid="stSidebar"] > div:first-child{padding-top:1.2rem}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]{color:#EAF2F8}
[data-testid="stSidebar"] hr{border-color:rgba(99,190,220,.18);margin:1.15rem 0}
[data-testid="stSidebar"] h3{font-family:'Space Grotesk',sans-serif!important;font-size:17px!important;letter-spacing:.2px}
[data-testid="stSidebar"] [role="radiogroup"]{gap:7px}
[data-testid="stSidebar"] [role="radiogroup"] label{
  position:relative;display:flex;align-items:center;
  padding:9px 12px!important;margin:0!important;border-radius:10px;
  border:1px solid transparent;background:rgba(255,255,255,.025);
  transition:all .22s ease;cursor:pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{
  background:linear-gradient(90deg,rgba(0,123,167,.18),rgba(255,45,149,.07));
  border-color:rgba(0,183,235,.24)
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){
  background:linear-gradient(100deg,rgba(0,123,167,.28),rgba(255,45,149,.12));
  border:1px solid rgba(41,190,235,.48);
  box-shadow:inset 3px 0 0 #19BCEB,0 5px 18px rgba(0,123,167,.12)
}
[data-testid="stSidebar"] [role="radiogroup"] label p{
  font-size:13px;font-weight:600;letter-spacing:.1px;color:#DCE7EF
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p{color:#FFFFFF}
[data-testid="stSidebar"] [role="radiogroup"] input{accent-color:#17BCEB}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#7F8C9A}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:var(--white)!important}
.brand{font:700 23px 'Space Grotesk';letter-spacing:1.5px}.brand span{color:#26B6E8}
.tagline{color:var(--muted);font-size:10px;letter-spacing:2px;text-transform:uppercase}
.kicker{color:#62D8F7;font-size:11px;font-weight:700;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px}
.hero{position:relative;overflow:hidden;padding:54px;border-radius:27px;border:1px solid #007ba75c;background:linear-gradient(112deg,#0a0a0af5 8%,#00415bd1 62%,#2a0a22c9);box-shadow:0 20px 80px #0008;animation:rise .8s ease both}
.hero-title{font:700 clamp(38px,5vw,64px)/1.06 'Space Grotesk';letter-spacing:-2px;color:white}
.accent{background:linear-gradient(90deg,#39C8F4,#007BA7 48%,#FF2D95);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero-copy{color:#C2CBD4;line-height:1.8;max-width:680px;margin-top:18px}
.card{background:linear-gradient(145deg,#171c26f5,#0d1118f5);border:1px solid #ffffff17;border-radius:17px;padding:20px;margin-bottom:12px}
.notice{padding:14px 17px;border-radius:12px;background:#007ba71a;border:1px solid #007ba74d;color:#C7DCE5;font-size:13px;line-height:1.65}
.metric{background:linear-gradient(145deg,#151a23,#0f1218);border:1px solid #007ba740;border-radius:16px;padding:20px}
.metric-label{color:#AAB4C0;font-size:10px;letter-spacing:1.5px;font-weight:700}
.metric-value{color:white;font:700 29px 'Space Grotesk';margin-top:10px}

/* Number-input steppers: neutral by default, reveal colour only on hover/focus. */
[data-testid="stNumberInput"] button{
  background:transparent!important;
  color:#AAB8C5!important;
  border:1px solid transparent!important;
  box-shadow:none!important;
  transition:background-color .18s ease,color .18s ease,border-color .18s ease!important;
}
[data-testid="stNumberInput"] button:first-of-type:hover,
[data-testid="stNumberInput"] button:first-of-type:focus-visible{
  background:#E58A24!important;
  color:#fff!important;
  border-color:#F2A64B!important;
}
[data-testid="stNumberInput"] button:last-of-type:hover,
[data-testid="stNumberInput"] button:last-of-type:focus-visible{
  background:#159B58!important;
  color:#fff!important;
  border-color:#36C879!important;
}
@media (prefers-reduced-motion:reduce){
  [data-testid="stNumberInput"] button{transition:none!important}
}

.stButton>button{background:linear-gradient(100deg,#007BA7,#0799C8);color:white;border:1px solid #40cdf55a;border-radius:11px;min-height:45px;font-weight:700;transition:.2s}
.stButton>button:hover{background:linear-gradient(100deg,#0799C8,#FF2D95);color:white;transform:translateY(-2px)}
div[data-testid="stForm"]{border:1px solid #ffffff19;border-radius:18px;padding:22px;background:#111318aa}
@keyframes rise{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@keyframes auroraDrift{
  0%{transform:translate3d(-2%,1%,0) scale(1.02) rotate(-2deg)}
  50%{transform:translate3d(2%,-1%,0) scale(1.08) rotate(1deg)}
  100%{transform:translate3d(-1%,-2%,0) scale(1.04) rotate(3deg)}
}
@keyframes softParticles{
  0%{transform:translate3d(0,0,0)}
  50%{transform:translate3d(0,-12px,0)}
  100%{transform:translate3d(0,0,0)}
}
/* Keep foreground forms readable and slightly glass-like. */
div[data-testid="stForm"]{background:rgba(12,17,24,.78);backdrop-filter:blur(12px);box-shadow:0 18px 55px rgba(0,0,0,.22)}
@media (prefers-reduced-motion: reduce){
  .stApp::before,.stApp::after,[data-testid="stAppViewContainer"]::before,.hero,.assessment-hero:before,.assessment-hero:after{animation:none!important}
  *,*::before,*::after{scroll-behavior:auto!important;transition:none!important}
}

[data-testid="stPlotlyChart"]{border:1px solid rgba(44,185,225,.18);border-radius:16px;
  background:rgba(10,16,26,.58);padding:7px;box-shadow:0 12px 35px rgba(0,0,0,.16)}
[data-testid="stDataFrame"]{border:1px solid rgba(44,185,225,.16);border-radius:12px;overflow:hidden}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(22,31,47,.94),rgba(10,15,24,.94));
  border:1px solid rgba(33,190,230,.20);border-radius:15px;padding:12px}
hr{border-color:#ffffff1a}

/* CINEMATIC + MEDITATION HYBRID WELCOME EXPERIENCE */
.cinematic-screen{
  min-height:calc(100svh - 13rem);position:relative;overflow:hidden;
  display:flex;align-items:center;justify-content:center;text-align:center;
  padding:24px 22px;border-radius:26px;
  background:
    radial-gradient(ellipse at 50% 48%,rgba(0,123,167,.19),transparent 43%),
    linear-gradient(145deg,rgba(4,9,17,.97),rgba(5,24,37,.96) 52%,rgba(18,7,23,.97));
  border:1px solid rgba(73,203,239,.24);
  box-shadow:0 28px 100px rgba(0,0,0,.42),inset 0 0 90px rgba(0,123,167,.08);
}
/* Slow aurora ribbons: atmospheric rather than fast game-like movement. */
.cinematic-screen .meditation-aurora{
  position:absolute;inset:-22%;pointer-events:none;z-index:0;opacity:.78;
  background:
    radial-gradient(ellipse 42% 24% at 18% 35%,rgba(0,198,220,.25),transparent 72%),
    radial-gradient(ellipse 38% 28% at 78% 30%,rgba(31,111,190,.22),transparent 74%),
    radial-gradient(ellipse 34% 25% at 64% 82%,rgba(255,45,149,.15),transparent 75%),
    radial-gradient(ellipse 30% 24% at 22% 78%,rgba(0,160,167,.18),transparent 74%);
  filter:blur(30px);animation:meditationFlow 32s ease-in-out infinite alternate;
}
.cinematic-screen .meditation-stars{
  position:absolute;inset:0;pointer-events:none;z-index:0;opacity:.42;
  background-image:radial-gradient(1px 1px at 14% 24%,#c5f4ff 98%,transparent),radial-gradient(1px 1px at 78% 18%,#c5f4ff 98%,transparent),radial-gradient(1px 1px at 65% 74%,#ffd4eb 98%,transparent),radial-gradient(1px 1px at 32% 82%,#c5f4ff 98%,transparent);
  background-size:300px 260px,390px 330px,440px 370px,360px 310px;
  animation:softParticles 42s ease-in-out infinite;
}
.cinematic-screen:before,.cinematic-screen:after{
  content:"";position:absolute;left:50%;top:48%;width:min(78vw,760px);aspect-ratio:1;
  border:1px solid rgba(73,203,239,.15);border-radius:50%;
  transform:translate(-50%,-50%);pointer-events:none;
  box-shadow:0 0 70px rgba(0,170,220,.08),inset 0 0 70px rgba(0,170,220,.05);
  animation:portalOrbit 18s linear infinite;
}
.cinematic-screen:after{
  width:min(60vw,570px);border-color:rgba(255,45,149,.16);
  animation-duration:26s;animation-direction:reverse;
}
.cinematic-grid{
  position:absolute;inset:0;opacity:.20;pointer-events:none;
  background-image:linear-gradient(rgba(93,203,232,.12) 1px,transparent 1px),
    linear-gradient(90deg,rgba(93,203,232,.12) 1px,transparent 1px);
  background-size:54px 54px;
  mask-image:radial-gradient(ellipse at center,#000 0%,transparent 72%);
}
.cinematic-content{position:relative;z-index:2;max-width:850px;animation:cinematicEnter 1.2s cubic-bezier(.2,.8,.2,1) both}
.cinematic-screen .cinematic-grid{z-index:1}
.cinematic-screen .cinematic-content{padding:8px 0}
.cinematic-symbol{
  width:82px;height:82px;margin:0 auto 24px;border-radius:25px;
  display:flex;align-items:center;justify-content:center;
  font:700 34px 'Space Grotesk';color:#E9FAFF;
  border:1px solid rgba(76,213,246,.55);
  background:linear-gradient(145deg,rgba(0,123,167,.28),rgba(255,45,149,.13));
  box-shadow:0 0 32px rgba(0,190,235,.23),inset 0 0 24px rgba(0,190,235,.12);
  animation:symbolFloat 5s ease-in-out infinite;
}
.cinematic-overline{font-size:10px;font-weight:800;letter-spacing:5px;text-transform:uppercase;color:#70DFF8}
.cinematic-title{
  margin:20px 0 12px;font:700 clamp(42px,7vw,82px)/.98 'Space Grotesk';
  letter-spacing:-3px;color:#F8FBFF;text-shadow:0 0 42px rgba(0,185,235,.17)
}
.cinematic-title span{
  display:block;margin-top:10px;
  background:linear-gradient(90deg,#57D8F7 5%,#159ACB 42%,#A05CB8 72%,#FF55AE 100%);
  -webkit-background-clip:text;background-clip:text;color:transparent
}
.cinematic-copy{max-width:620px;margin:22px auto 30px;color:#B9C9D6;font-size:15px;line-height:1.8}
.cinematic-line{width:150px;height:2px;margin:0 auto 28px;background:linear-gradient(90deg,transparent,#18C4EF,#FF2D95,transparent);box-shadow:0 0 18px rgba(0,190,235,.55);animation:linePulse 3s ease-in-out infinite}
.cinematic-foot{margin-top:24px;color:#74899A;font-size:10px;letter-spacing:2px;text-transform:uppercase}
.intro-mode [data-testid="stSidebar"],.intro-mode [data-testid="stHeader"],
.intro-mode [data-testid="stToolbar"],.intro-mode footer{display:none!important}
.intro-mode [data-testid="stMainBlockContainer"]{max-width:1500px;padding-top:1rem!important}
@keyframes cinematicEnter{from{opacity:0;transform:translateY(28px) scale(.975);filter:blur(8px)}to{opacity:1;transform:translateY(0) scale(1);filter:blur(0)}}
@keyframes portalOrbit{from{rotate:0deg;transform:translate(-50%,-50%) scale(.96)}to{rotate:360deg;transform:translate(-50%,-50%) scale(1.04)}}
@keyframes symbolFloat{0%,100%{transform:translateY(0);box-shadow:0 0 32px rgba(0,190,235,.23)}50%{transform:translateY(-8px);box-shadow:0 0 48px rgba(0,190,235,.38)}}
@keyframes linePulse{0%,100%{opacity:.55;transform:scaleX(.78)}50%{opacity:1;transform:scaleX(1.12)}}
@keyframes meditationFlow{0%{transform:translate3d(-2%,1%,0) rotate(-3deg) scale(1)}50%{transform:translate3d(2%,-1%,0) rotate(2deg) scale(1.08)}100%{transform:translate3d(-1%,-2%,0) rotate(4deg) scale(1.03)}}
/* Keep the enter action inside the first viewport on typical laptop screens. */
.intro-mode [data-testid="stMainBlockContainer"]{padding-bottom:.5rem!important}
.cinematic-screen + div{margin-top:-2px}
.cinematic-screen + div [data-testid="stButton"] button{min-height:48px;box-shadow:0 8px 28px rgba(0,153,205,.22);letter-spacing:.3px}
.cinematic-foot{margin-top:8px!important}
@media (max-height:760px){
 .cinematic-screen{min-height:calc(100svh - 10.5rem);padding:16px 18px}
 .cinematic-symbol{width:64px;height:64px;margin-bottom:15px}
 .cinematic-title{font-size:clamp(38px,6vw,68px);margin:13px 0 8px}
 .cinematic-copy{margin:14px auto 18px;font-size:14px}
 .cinematic-line{margin-bottom:18px}
}
@media (prefers-reduced-motion:reduce){
 .cinematic-screen:before,.cinematic-screen:after,.cinematic-content,.cinematic-symbol,.cinematic-line,.meditation-aurora,.meditation-stars{animation:none!important}
}

/* Assessment zone cards and visual labels */
.assessment-section-card{border:1px solid rgba(69,204,239,.22);border-radius:18px;background:linear-gradient(145deg,rgba(13,25,39,.9),rgba(10,15,26,.78));padding:16px;margin:14px 0 16px;box-shadow:0 12px 34px rgba(0,0,0,.16)}
.assessment-section-head{display:flex;align-items:center;gap:13px}
.assessment-icon{width:48px;height:48px;flex:0 0 48px;border-radius:15px;display:flex;align-items:center;justify-content:center;font-size:23px;background:linear-gradient(145deg,rgba(0,190,235,.20),rgba(139,92,246,.18));border:1px solid rgba(77,204,239,.28);box-shadow:0 0 24px rgba(0,190,235,.10)}
.assessment-section-title{font:700 19px 'Space Grotesk';color:#F1FAFF}
.assessment-section-copy{font-size:12px;color:#AFC3D3;margin-top:3px}
.assessment-field-label{font-size:13px;font-weight:700;color:#E7F7FF;margin:2px 0 5px}
.assessment-field-note{font-size:11px;color:#91A7B8;margin:-4px 0 8px;line-height:1.45}
.assessment-range{font-size:10px;letter-spacing:.8px;color:#6EDCF5;text-transform:uppercase;margin-bottom:4px}
.assessment-review{border:1px solid rgba(53,194,231,.22);border-radius:14px;padding:13px 15px;background:rgba(10,25,38,.72);margin:10px 0}
@media(max-width:700px){.assessment-hero{padding:23px 20px}.stage-row{grid-template-columns:1fr 1fr}.assessment-section-card{padding:13px}}

/* Number-input steppers: neutral by default, reveal colour only on hover/focus. */
[data-testid="stNumberInput"] button{
  background:transparent!important;
  color:#AAB8C5!important;
  border:1px solid transparent!important;
  box-shadow:none!important;
  transition:background-color .18s ease,color .18s ease,border-color .18s ease!important;
}
[data-testid="stNumberInput"] button:first-of-type:hover,
[data-testid="stNumberInput"] button:first-of-type:focus-visible{
  background:#E58A24!important;
  color:#fff!important;
  border-color:#F2A64B!important;
}
[data-testid="stNumberInput"] button:last-of-type:hover,
[data-testid="stNumberInput"] button:last-of-type:focus-visible{
  background:#159B58!important;
  color:#fff!important;
  border-color:#36C879!important;
}
@media (prefers-reduced-motion:reduce){
  [data-testid="stNumberInput"] button{transition:none!important}
}

</style>
""", unsafe_allow_html=True)

# ----------------------------- SESSION STATE -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "mode" not in st.session_state:
    st.session_state.mode = "Student"
if "assessment_result" not in st.session_state:
    st.session_state.assessment_result = None
if "assessment_history" not in st.session_state:
    st.session_state.assessment_history = []
if "intro_seen" not in st.session_state:
    st.session_state.intro_seen = False

# Navigation callbacks: commit the selected destination before Streamlit reruns.
def _sync_page_navigation():
    st.session_state.page = st.session_state.page_navigation

def _sync_portal_mode():
    # Entering Counselor portal should open its dashboard immediately.
    if st.session_state.mode == "Counselor":
        destination = "Counselor Dashboard"
    else:
        destination = "Home"
    st.session_state.page = destination
    st.session_state.page_navigation = destination

def _open_student_portal():
    # Widget state is changed safely inside the button callback,
    # before Streamlit instantiates the portal radio on the next rerun.
    st.session_state.mode = "Student"
    st.session_state.page = "My Dashboard"
    st.session_state.page_navigation = "My Dashboard"

def _open_counselor_portal():
    st.session_state.mode = "Counselor"
    st.session_state.page = "Counselor Dashboard"
    st.session_state.page_navigation = "Counselor Dashboard"

# ----------------------------- SIDEBAR -----------------------------
if not st.session_state.intro_seen:
    st.markdown("<style>[data-testid='stSidebar'],[data-testid='stHeader'],[data-testid='stToolbar'],footer{display:none!important} body{overflow:hidden} [data-testid='stMainBlockContainer']{max-width:1500px;padding-top:1rem!important}</style>", unsafe_allow_html=True)
    st.markdown("<div class=\"intro-mode\"></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('''
    <div style="padding:15px 13px 19px;border:1px solid rgba(0,183,235,.22);
      border-radius:16px;background:linear-gradient(135deg,rgba(0,123,167,.13),rgba(255,45,149,.055));
      box-shadow:0 10px 30px rgba(0,0,0,.18);">
      <div style="font:700 22px 'Space Grotesk';letter-spacing:1.2px;color:#F5F7FA;">
        WELLQUEST <span style="color:#26B6E8;">AI</span>
      </div>
      <div style="color:#AAB8C7;font-size:9px;letter-spacing:2.1px;margin-top:5px;text-transform:uppercase;">
        Your wellbeing. Your journey.
      </div>
      <div style="height:2px;width:54px;margin-top:13px;border-radius:4px;
        background:linear-gradient(90deg,#007BA7,#FF2D95);"></div>
    </div>
    ''', unsafe_allow_html=True)
    st.write("")
    st.divider()
    st.markdown("### Workspace")
    st.radio(
        "Choose portal",
        ["Student", "Counselor"],
        key="mode",
        on_change=_sync_portal_mode,
    )
    mode = st.session_state.mode
    st.divider()
    pages = (["Home", "My Dashboard", "Wellbeing Assessment", "My Insights", "Growth Journey", "Support Hub"]
             if mode == "Student" else
             ["Home", "Counselor Dashboard", "Model Analytics", "Student Reviews", "Reports"])

    # Keep the radio widget and app page in sync, including navigation from page buttons.
    if st.session_state.page not in pages:
        st.session_state.page = "Home"
    if st.session_state.get("page_navigation") not in pages or st.session_state.page_navigation != st.session_state.page:
        st.session_state.page_navigation = st.session_state.page

    st.radio(
        "Navigation",
        pages,
        key="page_navigation",
        on_change=_sync_page_navigation,
    )
    st.divider()
    st.caption("WELLQUEST AI")
    st.caption("Academic prototype | 2026")
    st.write("")
    if st.button("Replay cinematic welcome", use_container_width=True):
        st.session_state.intro_seen = False
        st.rerun()

# ----------------------------- CINEMATIC WELCOME -----------------------------
if not st.session_state.intro_seen:
    st.markdown("""
    <section class="cinematic-screen">
      <div class="meditation-aurora"></div>
      <div class="meditation-stars"></div>
      <div class="cinematic-grid"></div>
      <div class="cinematic-content">
        <div class="cinematic-symbol">W<span style="color:#32C7EE">.</span></div>
        <div class="cinematic-overline">A NEW PERSPECTIVE ON STUDENT WELLBEING</div>
        <div class="cinematic-title">WELLQUEST<span>AI</span></div>
        <div class="cinematic-line"></div>
        <div class="cinematic-copy">
          Understand the patterns. Discover meaningful insights.
          Make room for better conversations and human-centred support.
        </div>
        <div class="cinematic-foot">A CALMER WAY TO SEE THE BIGGER PICTURE</div>
      </div>
    </section>
    """, unsafe_allow_html=True)
    _, enter_col, _ = st.columns([1, 1.15, 1])
    with enter_col:
        if st.button("ENTER WELLQUEST  →", use_container_width=True):
            st.session_state.intro_seen = True
            st.session_state.page = "Home"
            st.rerun()
    st.markdown('<div class="cinematic-foot" style="text-align:center">YOUR WELLBEING. YOUR JOURNEY.</div>', unsafe_allow_html=True)

# ----------------------------- HOME -----------------------------
elif st.session_state.page == "Home":
    st.markdown("""
    <section class="hero">
      <div class="kicker">INTELLIGENCE MEETS WELLBEING</div>
      <div class="hero-title">Understand patterns.<br><span class="accent">Build better wellbeing.</span></div>
      <div class="hero-copy">A student-focused AI platform that explores patterns in academic and wellbeing data. Turn structured insights into thoughtful conversations and more informed human support.</div>
    </section>
    """, unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    with a:
        st.button(
            "Explore Student Portal →",
            use_container_width=True,
            on_click=_open_student_portal,
        )
    with b:
        st.button(
            "Open Counselor Portal →",
            use_container_width=True,
            on_click=_open_counselor_portal,
        )
    st.write("")
    st.markdown('<div class="kicker">THE WELLQUEST APPROACH</div><h2>A clearer view of student wellbeing</h2>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, n, title, desc in zip(cols, ["01", "02", "03"],
        ["Understand patterns", "Discover insights", "Encourage support"],
        ["Explore academic and wellbeing dataset features.",
         "View transparent model outputs, not diagnoses.",
         "Support thoughtful, human-led conversations."]):
        with col:
            st.markdown(f'<div class="card"><div class="kicker">{n} / WELLQUEST</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
    st.caption("Academic screening-support prototype. Not a diagnostic tool.")

# ----------------------------- STUDENT DASHBOARD -----------------------------
elif st.session_state.page == "My Dashboard":
    result = st.session_state.assessment_result
    history = st.session_state.assessment_history

    st.markdown('<div class="kicker">STUDENT SPACE  /  YOUR WELLQUEST OVERVIEW</div>', unsafe_allow_html=True)
    st.title("Your Dashboard")
    st.write("A personal overview of your latest check-in and the information you have entered.")

    if result is None:
        st.markdown("""
        <div class="assessment-hero">
          <div class="assessment-hero-inner">
            <div class="assessment-eyebrow">YOUR JOURNEY STARTS HERE</div>
            <div class="assessment-heading">Make your first<br><span class="accent">wellbeing check-in.</span></div>
            <div class="assessment-sub">Complete the research-based assessment to populate your dashboard with your latest model output and entered indicator values.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.info("No assessment has been completed in this session yet.")
        if st.button("Start wellbeing assessment →", type="primary", use_container_width=True):
            st.session_state.page = "Wellbeing Assessment"
            st.rerun()
    else:
        probs = result.get("probabilities")
        assessed_at = result.get("timestamp", "Time not recorded")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric"><div class="metric-label">LATEST DATASET CLASS</div><div class="metric-value">{result["class"]}</div><p>Model-predicted label</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric"><div class="metric-label">ASSESSMENT STATUS</div><div class="metric-value">Complete</div><p>{assessed_at}</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric"><div class="metric-label">CHECK-INS THIS SESSION</div><div class="metric-value">{len(history)}</div><p>History held in this browser session</p></div>', unsafe_allow_html=True)

        st.markdown("### Latest model output")
        st.markdown('<div class="notice">Class 0, 1 and 2 are shown exactly as dataset labels. Their wellbeing meanings have not been verified from the source codebook. Model probabilities are not clinical risk probabilities or diagnoses.</div>', unsafe_allow_html=True)

        if probs:
            prob_labels = [f"Class {i}" for i in range(len(probs))]
            prob_values = [round(float(p) * 100, 2) for p in probs]
            chart_col, metric_col = st.columns([1.05, 1.5])
            with chart_col:
                st.plotly_chart(make_donut(prob_labels, prob_values, "Prediction distribution"), use_container_width=True, config=PLOT_CONFIG)
            with metric_col:
                st.markdown("#### Model probability estimates")
                pcols = st.columns(len(probs))
                for i, col in enumerate(pcols):
                    with col:
                        st.markdown(f'<div class="metric"><div class="metric-label">CLASS {i}</div><div class="metric-value">{float(probs[i])*100:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("### Your entered indicators")
        st.caption("These are the values submitted to the model. The original questionnaire scale meanings should be checked against the dataset codebook before interpreting them.")
        value_rows = []
        groups = [
            ("Personal wellbeing", ["anxiety_level", "self_esteem", "mental_health_history", "depression"]),
            ("Physical and environmental", ["headache", "blood_pressure", "sleep_quality", "breathing_problem", "noise_level", "living_conditions", "safety", "basic_needs"]),
            ("Academic and social", ["academic_performance", "study_load", "teacher_student_relationship", "future_career_concerns", "social_support", "peer_pressure", "extracurricular_activities", "bullying"]),
        ]
        for group_name, features in groups:
            for feature in features:
                value_rows.append({
                    "Category": group_name,
                    "Indicator": feature.replace("_", " ").title(),
                    "Entered value": result.get("values", {}).get(feature, "—"),
                    "Dataset range": f"{RANGES[feature][0]}–{RANGES[feature][1]}",
                })
        st.dataframe(pd.DataFrame(value_rows), use_container_width=True, hide_index=True)

        st.markdown("### AI-generated visual profile")
        st.caption("Radar values show each entered value's position inside the observed dataset range. They are not a personal wellbeing score.")
        radar_groups = {
            "Personal wellbeing": ["anxiety_level", "self_esteem", "depression"],
            "Physical & environment": ["sleep_quality", "headache", "breathing_problem", "safety"],
            "Academic & social": ["study_load", "academic_performance", "social_support", "peer_pressure"],
        }
        rcols = st.columns(3)
        for col, (group_name, feats) in zip(rcols, radar_groups.items()):
            with col:
                st.plotly_chart(make_radar(range_positions(result.get("values", {}), feats), group_name), use_container_width=True, config=PLOT_CONFIG)

        if len(history) > 1:
            hist_df = pd.DataFrame(history)
            hist_df["Check-in"] = range(1, len(hist_df)+1)
            st.markdown("### Session pattern timeline")
            fig_hist = px.line(hist_df, x="Check-in", y="class", markers=True, title="Predicted dataset class across this session")
            fig_hist.update_traces(line_color="#20C9EA", marker=dict(size=10, color="#FF4FA3"))
            fig_hist.update_layout(**PLOT_LAYOUT, height=280, yaxis_title="Dataset class label")
            st.plotly_chart(fig_hist, use_container_width=True, config=PLOT_CONFIG)
            st.caption("A change in class label is a model output change, not evidence of a clinical improvement or decline.")

        st.write("")
        left, right = st.columns(2)
        with left:
            if st.button("Retake assessment →", use_container_width=True):
                st.session_state.page = "Wellbeing Assessment"
                st.rerun()
        with right:
            export_df = pd.DataFrame([{
                "timestamp": result.get("timestamp", ""),
                "predicted_dataset_class": result.get("class", ""),
                **result.get("values", {}),
                **({f"class_{i}_model_estimate": float(p) for i, p in enumerate(probs)} if probs else {}),
            }])
            st.download_button(
                "Download latest check-in (CSV)",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="wellquest_latest_checkin.csv",
                mime="text/csv",
                use_container_width=True,
            )

        if len(history) > 1:
            st.markdown("### Previous check-ins")
            history_rows = []
            for item in reversed(history):
                history_rows.append({
                    "Date and time": item.get("timestamp", ""),
                    "Predicted dataset class": item.get("class", ""),
                })
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)

# ----------------------------- ACTUAL ASSESSMENT -----------------------------
elif st.session_state.page == "Wellbeing Assessment":
    st.markdown("""
    <section class="assessment-hero">
      <div class="assessment-hero-inner">
        <div class="assessment-eyebrow">WELLQUEST AI  /  INTERACTIVE CHECK-IN</div>
        <div class="assessment-heading">Your wellbeing has<br><span class="accent">many dimensions.</span></div>
        <div class="assessment-sub">Explore 20 research-dataset indicators through four clearly organised areas. Each card explains the variable and shows the numerical range found in the dataset.</div>
        <span class="assessment-chip">✦ 20 indicators</span><span class="assessment-chip">◈ 4 themed zones</span><span class="assessment-chip">⌁ Dual-model analysis</span>
      </div>
    </section>
    <div class="stage-row" style="grid-template-columns:repeat(4,1fr)">
      <div class="stage active"><b>01 · Mind</b>Personal wellbeing</div>
      <div class="stage active"><b>02 · Body</b>Physical & environment</div>
      <div class="stage active"><b>03 · Study</b>Academic experience</div>
      <div class="stage active"><b>04 · People</b>Social environment</div>
    </div>
    <div class="notice">Important: these are observed numerical ranges from the research dataset, not verified questionnaire answer scales. Do not guess a number. If you are unsure what a coded value means, leave the default unchanged until the AI-Guided Quiz is added in Phase 2. This is not a clinical assessment or diagnosis.</div>
    """, unsafe_allow_html=True)

    motion_paused = st.toggle("Pause ambient background motion", value=False)
    if motion_paused:
        st.markdown('<style>.stApp::before,.stApp::after,[data-testid="stAppViewContainer"]::before{animation-play-state:paused!important}</style>', unsafe_allow_html=True)

    field_info = {
        "anxiety_level": ("◉", "Anxiety level", "Dataset indicator associated with reported anxiety."),
        "self_esteem": ("✧", "Self-esteem", "Dataset indicator associated with self-esteem."),
        "mental_health_history": ("♡", "Mental-health history", "Binary dataset field. Confirm its exact coding from the source codebook."),
        "depression": ("☾", "Depression", "Dataset indicator associated with depression-related responses."),
        "headache": ("✚", "Headache", "Dataset indicator relating to headache; exact anchor wording is not verified."),
        "blood_pressure": ("♥", "Blood pressure", "Dataset-coded category, not a medical blood-pressure reading."),
        "sleep_quality": ("☾", "Sleep quality", "Dataset indicator relating to sleep quality."),
        "breathing_problem": ("≈", "Breathing problems", "Dataset indicator relating to breathing difficulties."),
        "noise_level": ("♫", "Noise level", "Dataset indicator relating to noise in the surrounding environment."),
        "living_conditions": ("⌂", "Living conditions", "Dataset indicator relating to the student's living environment."),
        "safety": ("⬡", "Safety", "Dataset indicator relating to perceived safety."),
        "basic_needs": ("◇", "Basic needs", "Dataset indicator relating to access to basic needs."),
        "academic_performance": ("▤", "Academic performance", "Dataset indicator relating to academic performance."),
        "study_load": ("▦", "Study load", "Dataset indicator relating to study workload."),
        "teacher_student_relationship": ("▣", "Teacher–student relationship", "Dataset indicator relating to the student–teacher relationship."),
        "future_career_concerns": ("↗", "Future career concerns", "Dataset indicator relating to career-related concerns."),
        "social_support": ("◎", "Social support", "Dataset indicator relating to available social support."),
        "peer_pressure": ("◌", "Peer pressure", "Dataset indicator relating to peer influence or pressure."),
        "extracurricular_activities": ("★", "Extracurricular activities", "Dataset indicator relating to extracurricular participation."),
        "bullying": ("⚑", "Bullying", "Dataset indicator relating to bullying experiences."),
    }

    sections = [
        ("01", "◉", "Mind & emotions", "Personal wellbeing indicators", ["anxiety_level", "self_esteem", "mental_health_history", "depression"]),
        ("02", "♥", "Body & surroundings", "Physical and everyday environment", ["headache", "blood_pressure", "sleep_quality", "breathing_problem", "noise_level", "living_conditions", "safety", "basic_needs"]),
        ("03", "▤", "Academic life", "Learning, workload and future concerns", ["academic_performance", "study_load", "teacher_student_relationship", "future_career_concerns"]),
        ("04", "◎", "People & connection", "Support and social environment", ["social_support", "peer_pressure", "extracurricular_activities", "bullying"]),
    ]

    if not MODEL_PATH.exists():
        st.error(f"Trained model file not found: {MODEL_PATH}")
        st.info("Check that models/logistic_regression.joblib exists in your WELLQUEST_AI project folder.")
    else:
        with st.form("wellbeing_assessment_form"):
            st.markdown("### Assessment console")
            st.caption("Enter dataset-coded values only when you know the source coding. These controls retain compatibility with the trained model.")
            values = {}
            for section_no, icon, section_title, section_copy, features in sections:
                st.markdown(f"""
                <div class="assessment-section-card">
                  <div class="assessment-section-head">
                    <div class="assessment-icon">{icon}</div>
                    <div>
                      <div class="assessment-range">ZONE {section_no} · {len(features)} INDICATORS</div>
                      <div class="assessment-section-title">{section_title}</div>
                      <div class="assessment-section-copy">{section_copy}</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                grid = st.columns(2)
                for i, feature in enumerate(features):
                    low, high = RANGES[feature]
                    default = int(round((low + high) / 2))
                    symbol, label, description = field_info[feature]
                    with grid[i % 2]:
                        st.markdown(f"""
                        <div class="assessment-field-label">{symbol} &nbsp; {label}</div>
                        <div class="assessment-field-note">{description}</div>
                        <div class="assessment-range">Observed dataset range: {low} – {high}</div>
                        """, unsafe_allow_html=True)
                        if feature == "mental_health_history":
                            values[feature] = st.selectbox(
                                "Select dataset code", [0, 1], index=0,
                                key=f"assessment_{feature}",
                                help="Confirm the meaning of 0 and 1 from the original dataset codebook."
                            )
                        else:
                            values[feature] = st.number_input(
                                "Dataset-coded value", min_value=int(low), max_value=int(high),
                                value=default, step=1, key=f"assessment_{feature}",
                                help=f"Observed values range from {low} to {high}; this does not define what each score means."
                            )

            st.markdown("""
            <div class="assessment-review">
              <b>FINAL REVIEW</b><br>
              Your 20 selected dataset-coded values will be sent to the trained Logistic Regression model. The model predicts a dataset class, not a clinical diagnosis.
            </div>
            """, unsafe_allow_html=True)
            submitted = st.form_submit_button("✦  RUN WELLQUEST AI ANALYSIS", use_container_width=True)

        if submitted:
            try:
                model = joblib.load(MODEL_PATH)
                input_df = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
                prediction = int(model.predict(input_df)[0])
                probabilities = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None
                result_record = {
                    "class": prediction,
                    "probabilities": [float(x) for x in probabilities] if probabilities is not None else None,
                    "values": values,
                    "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                }
                st.session_state.assessment_result = result_record
                st.session_state.assessment_history.append({
                    "class": prediction,
                    "timestamp": result_record["timestamp"],
                    "probabilities": result_record["probabilities"],
                    "values": dict(values),
                })
                st.rerun()
            except Exception as exc:
                st.error(f"Could not generate the prediction: {exc}")

    if st.session_state.assessment_result:
        result = st.session_state.assessment_result
        st.divider()
        st.markdown('<div class="kicker">MODEL OUTPUT</div>', unsafe_allow_html=True)
        st.subheader(f"Predicted dataset class: {result['class']}")
        st.markdown('<div class="notice">This is a prediction of the research dataset label, not a diagnosis or a validated measure of personal wellbeing. It may be inaccurate and should not be used alone to make decisions.</div>', unsafe_allow_html=True)
        if result["probabilities"] is not None:
            st.markdown("#### Model probability estimates")
            prob_cols = st.columns(3)
            for i, col in enumerate(prob_cols):
                with col:
                    st.markdown(f'<div class="metric"><div class="metric-label">CLASS {i}</div><div class="metric-value">{result["probabilities"][i]*100:.1f}%</div></div>', unsafe_allow_html=True)
        if st.button("Clear this assessment"):
            st.session_state.assessment_result = None
            st.rerun()

# ----------------------------- AI INSIGHTS LAB -----------------------------
elif st.session_state.page == "My Insights":
    result = st.session_state.assessment_result
    st.markdown('<div class="kicker">WELLQUEST AI  /  EXPLAINABLE INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("My Insights")
    st.write("Explore how the trained models interpret your submitted indicators, compare model outputs, and test hypothetical changes.")

    if result is None:
        st.markdown("""
        <div class="assessment-hero"><div class="assessment-hero-inner">
        <div class="assessment-eyebrow">AI INSIGHTS ARE WAITING</div>
        <div class="assessment-heading">Start with a<br><span class="accent">wellbeing check-in.</span></div>
        <div class="assessment-sub">Complete an assessment first. WELLQUEST will then run model comparison, feature analysis and a what-if simulation using your submitted values.</div>
        </div></div>
        """, unsafe_allow_html=True)
        if st.button("Go to Wellbeing Assessment →", type="primary", use_container_width=True):
            st.session_state.page = "Wellbeing Assessment"
            st.rerun()
    else:
        input_values = result.get("values", {})
        input_df = pd.DataFrame([[input_values.get(f, 0) for f in FEATURES]], columns=FEATURES)
        rf_path = BASE_DIR / "models" / "random_forest.joblib"
        lr_path = MODEL_PATH
        lr_model = joblib.load(lr_path) if lr_path.exists() else None
        rf_model = joblib.load(rf_path) if rf_path.exists() else None

        st.markdown('<div class="notice">AI transparency: these are predictions from trained models using a secondary research dataset. Dataset class labels have not been semantically verified. Model estimates are not diagnoses, clinical risk scores, or causal explanations.</div>', unsafe_allow_html=True)

        # Quick-glance insight strip: derived directly from the latest submitted record.
        quick_cols = st.columns(4)
        latest_class = result.get("class", "Not available")
        quick_items = [
            ("LATEST DATASET CLASS", f"{latest_class}"),
            ("INPUT INDICATORS", str(len([f for f in FEATURES if f in input_values])) + " / 20"),
            ("MODELS AVAILABLE", str(sum([(MODEL_PATH.exists()), ((BASE_DIR / "models" / "random_forest.joblib").exists())]))),
            ("RECORD CONTEXT", "Session assessment"),
        ]
        for col, (label, value) in zip(quick_cols, quick_items):
            with col:
                st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:25px">{value}</div></div>', unsafe_allow_html=True)

        # 1) Model comparison: both trained estimators evaluate the same submitted row.
        st.markdown("### 01 / Dual-model intelligence")
        comparison = []
        model_objects = []
        for model_name, model_obj in [("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
            if model_obj is not None:
                pred = int(model_obj.predict(input_df)[0])
                probs = model_obj.predict_proba(input_df)[0] if hasattr(model_obj, "predict_proba") else None
                confidence = float(max(probs) * 100) if probs is not None else None
                comparison.append({"Model": model_name, "Predicted dataset class": pred,
                                   "Highest class estimate": f"{confidence:.1f}%" if confidence is not None else "Not available"})
                model_objects.append((model_name, model_obj, pred, probs))
        if comparison:
            c1, c2 = st.columns(2)
            for idx, col in enumerate([c1, c2]):
                if idx < len(comparison):
                    row = comparison[idx]
                    with col:
                        st.markdown(f'<div class="metric"><div class="metric-label">{row["Model"].upper()}</div><div class="metric-value">CLASS {row["Predicted dataset class"]}</div><p>Highest class estimate: {row["Highest class estimate"]}</p></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
            probability_frames = []
            for model_name, model_obj in [("Logistic Regression", lr_model), ("Random Forest", rf_model)]:
                frame = model_probability_frame(model_obj, input_df)
                if frame is not None:
                    frame["Model"] = model_name
                    probability_frames.append(frame)
            if probability_frames:
                prob_compare = pd.concat(probability_frames, ignore_index=True)
                compare_fig = px.bar(prob_compare, x="Dataset class", y="Estimate (%)", color="Model", barmode="group",
                    title="Two-model probability comparison", color_discrete_sequence=["#13C8E8", "#FF4FA3"])
                compare_fig.update_layout(**PLOT_LAYOUT, height=330, yaxis_range=[0,100])
                st.plotly_chart(compare_fig, use_container_width=True, config=PLOT_CONFIG)
            if len(comparison) == 2:
                if comparison[0]["Predicted dataset class"] == comparison[1]["Predicted dataset class"]:
                    st.success("Both trained models agree on the predicted dataset class for this input.")
                else:
                    st.warning("The models disagree on this input. Treat the result cautiously and review the model limitations.")
        else:
            st.error("No trained model files were found. Check the models folder.")

        # 2) Global feature importance from the trained random forest.
        st.markdown("### 02 / Explainable AI — influential features")
        if rf_model is not None and hasattr(rf_model, "feature_importances_"):
            importance = pd.DataFrame({"Feature": FEATURES, "Importance": rf_model.feature_importances_})
            importance = importance.sort_values("Importance", ascending=False).head(10)
            st.caption("Global Random Forest feature importance across the training process. This is not a personalised causal explanation for one student.")
            importance_fig = px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                title="Top 10 global feature importance", color="Importance",
                color_continuous_scale=[[0,"#164E63"],[.55,"#13C8E8"],[1,"#FF4FA3"]])
            importance_fig.update_layout(**PLOT_LAYOUT, height=430, coloraxis_showscale=False,
                xaxis_title="Relative importance", yaxis_title="")
            st.plotly_chart(importance_fig, use_container_width=True, config=PLOT_CONFIG)
            st.dataframe(importance.assign(Importance=importance["Importance"].round(4)), use_container_width=True, hide_index=True)
        else:
            st.info("Feature importance is unavailable for the loaded Random Forest model.")

        # 3) Pattern review: descriptive grouping and transparent threshold flags.
        st.markdown("### 03 / Personal pattern review")
        categories = {
            "Personal wellbeing": ["anxiety_level", "self_esteem", "mental_health_history", "depression"],
            "Physical and environment": ["headache", "blood_pressure", "sleep_quality", "breathing_problem", "noise_level", "living_conditions", "safety", "basic_needs"],
            "Academic and social": ["academic_performance", "study_load", "teacher_student_relationship", "future_career_concerns", "social_support", "peer_pressure", "extracurricular_activities", "bullying"],
        }
        group_rows = []
        for group_name, feats in categories.items():
            # Average of each feature's position within its observed dataset range.
            scaled = []
            for feat in feats:
                lo, hi = RANGES[feat]
                span = hi - lo
                scaled.append((float(input_values.get(feat, lo)) - lo) / span if span else 0)
            group_rows.append({"Indicator group": group_name, "Average range position (%)": round(sum(scaled) / len(scaled) * 100, 1)})
        group_df = pd.DataFrame(group_rows)
        group_fig = px.bar(group_df, x="Indicator group", y="Average range position (%)", color="Indicator group",
            title="Indicator group profile", color_discrete_sequence=["#13C8E8", "#8B5CF6", "#FF4FA3"])
        group_fig.update_layout(**PLOT_LAYOUT, height=320, showlegend=False, yaxis_range=[0,100], xaxis_title="", yaxis_title="Range position (%)")
        st.plotly_chart(group_fig, use_container_width=True, config=PLOT_CONFIG)

        st.markdown("#### Personal indicator radar")
        radar_features = ["anxiety_level", "self_esteem", "depression", "sleep_quality", "study_load",
                          "social_support", "peer_pressure", "future_career_concerns"]
        radar_values = {}
        for feat in radar_features:
            lo, hi = RANGES[feat]
            raw = float(input_values.get(feat, lo))
            radar_values[feat] = round((raw - lo) / (hi - lo) * 100, 1) if hi > lo else 0
        radar_col, detail_col = st.columns([1.15, 1])
        with radar_col:
            radar_fig = make_radar(radar_values, "Submitted indicators · observed-range position")
            st.plotly_chart(radar_fig, use_container_width=True, config=PLOT_CONFIG)
        with detail_col:
            radar_table = pd.DataFrame([
                {"Indicator": feat.replace("_", " ").title(),
                 "Submitted value": input_values.get(feat, RANGES[feat][0]),
                 "Observed range": f"{RANGES[feat][0]}–{RANGES[feat][1]}",
                 "Range position (%)": radar_values[feat]}
                for feat in radar_features
            ])
            st.dataframe(radar_table, use_container_width=True, hide_index=True)
            st.caption("Radar values are normalised positions within observed dataset ranges. They are not good/bad scores and do not share a common clinical scale.")
        st.caption("Range position is a descriptive normalisation of dataset-coded inputs, not a wellbeing score or judgement.")

        # Transparent pattern flags and non-clinical suggestions.
        observations = []
        suggestions = []
        for feat in ["anxiety_level", "depression", "study_load", "future_career_concerns", "peer_pressure", "bullying", "sleep_quality", "social_support", "self_esteem"]:
            if feat not in input_values:
                continue
            lo, hi = RANGES[feat]
            v = float(input_values[feat])
            position = (v - lo) / (hi - lo) if hi > lo else 0
            label = feat.replace("_", " ").title()
            if feat in ["sleep_quality", "social_support", "self_esteem"]:
                flagged = position <= 0.35
            else:
                flagged = position >= 0.65
            if flagged:
                observations.append(f"{label} is toward the {'lower' if feat in ['sleep_quality','social_support','self_esteem'] else 'upper'} end of its observed dataset range ({v:g} on a {lo}–{hi} range).")
                if feat == "study_load": suggestions.append("Consider breaking study tasks into shorter, planned blocks and reviewing workload with an academic mentor.")
                elif feat in ["anxiety_level", "depression"]: suggestions.append("If this reflects how you feel, consider talking with a trusted person or qualified campus support professional.")
                elif feat == "sleep_quality": suggestions.append("Consider reviewing your sleep routine and seeking support if sleep difficulties persist.")
                elif feat in ["social_support", "self_esteem"]: suggestions.append("Consider reconnecting with a trusted friend, mentor or student support service.")
                elif feat in ["bullying", "peer_pressure"]: suggestions.append("If this reflects an unsafe or uncomfortable situation, consider contacting a trusted person or appropriate campus support.")
                elif feat == "future_career_concerns": suggestions.append("A short career-planning conversation with a faculty mentor may help clarify next steps.")
        if observations:
            for item in observations:
                st.markdown(f"- {item}")
        else:
            st.info("No indicator crossed the simple review thresholds used in this prototype. This does not mean that no support is needed.")
        if suggestions:
            st.markdown("#### Suggested next steps")
            for item in dict.fromkeys(suggestions):
                st.markdown(f"- {item}")
        st.caption("These observations are generated by transparent prototype rules, not by a generative AI service or a validated clinical framework.")

        # 4) What-if simulator: alter one feature and rerun both trained models.
        st.markdown("### 04 / What-if AI simulator")
        st.write("Change one input temporarily to see whether the trained model predictions change. This is an experiment, not a recommendation to alter real-life circumstances.")
        chosen_feature = st.selectbox("Choose an indicator to simulate", FEATURES, format_func=lambda x: x.replace("_", " ").title())
        low, high = RANGES[chosen_feature]
        current_value = int(input_values.get(chosen_feature, low))
        simulated_value = st.slider("Simulated value", min_value=int(low), max_value=int(high), value=current_value, step=1, key="whatif_value")
        sim_values = dict(input_values)
        sim_values[chosen_feature] = simulated_value
        sim_df = pd.DataFrame([[sim_values.get(f, 0) for f in FEATURES]], columns=FEATURES)
        if st.button("Run what-if simulation", use_container_width=True):
            sim_cols = st.columns(2)
            sim_chart_rows = []
            for idx, (model_name, model_obj) in enumerate([("Logistic Regression", lr_model), ("Random Forest", rf_model)]):
                if model_obj is not None:
                    before = int(model_obj.predict(input_df)[0])
                    after = int(model_obj.predict(sim_df)[0])
                    with sim_cols[idx]:
                        st.markdown(f"**{model_name}**")
                        st.metric("Simulated dataset class", after, delta=f"{after-before:+d} vs original" if after != before else "No class change")
                    if hasattr(model_obj, "predict_proba"):
                        before_p = model_obj.predict_proba(input_df)[0]
                        after_p = model_obj.predict_proba(sim_df)[0]
                        classes = getattr(model_obj, "classes_", range(len(before_p)))
                        for cl, bp, ap in zip(classes, before_p, after_p):
                            sim_chart_rows.extend([
                                {"Model": model_name, "Dataset class": f"Class {cl}", "Scenario": "Original", "Estimate (%)": float(bp)*100},
                                {"Model": model_name, "Dataset class": f"Class {cl}", "Scenario": "What-if", "Estimate (%)": float(ap)*100},
                            ])
            if sim_chart_rows:
                sim_fig = px.bar(pd.DataFrame(sim_chart_rows), x="Dataset class", y="Estimate (%)", color="Scenario", barmode="group",
                    facet_col="Model", title="Original vs what-if model estimates", color_discrete_sequence=["#13C8E8", "#FF4FA3"])
                sim_fig.update_layout(**PLOT_LAYOUT, height=370, yaxis_range=[0,100])
                st.plotly_chart(sim_fig, use_container_width=True, config=PLOT_CONFIG)
            st.caption("Only the selected input was changed; all other submitted values were held constant. This is a model sensitivity experiment, not a recommended life change.")

        # 5) Explain how the AI pipeline works for academic demonstration.
        with st.expander("How WELLQUEST AI works (for your AIML presentation)"):
            st.markdown("""
            1. **Input layer:** collects 20 structured indicators.
            2. **Pre-processing:** arranges values in the same feature order used during model training.
            3. **Inference:** Logistic Regression and Random Forest generate dataset-class predictions.
            4. **Probability output:** where supported, each model returns class probability estimates.
            5. **Explainability:** Random Forest feature importance shows which variables were influential across training.
            6. **Counterfactual-style experiment:** the what-if simulator changes one feature and reruns inference.
            7. **Human-centred support:** transparent observations are shown as prompts for reflection, never as diagnosis.
            """)

# ----------------------------- COUNSELOR MODEL ANALYTICS -----------------------------
elif st.session_state.page == "Model Analytics":
    st.markdown('<div class="kicker">MODEL INTELLIGENCE  /  SYSTEM PERFORMANCE</div>', unsafe_allow_html=True)
    st.title("Model Analytics")
    st.markdown("""
    <section class="assessment-hero">
      <div class="assessment-hero-inner">
        <div class="assessment-eyebrow">WELLQUEST AI · MODEL LAB</div>
        <div class="assessment-heading">Inside the<br><span class="accent">intelligence engine.</span></div>
        <div class="assessment-sub">Explore the trained model artifacts, compare their recorded holdout metrics, and inspect global feature importance. These are preliminary academic results—not proof of clinical validity or real-world performance.</div>
        <span class="assessment-chip">20 structured inputs</span>
        <span class="assessment-chip">2 trained models</span>
        <span class="assessment-chip">Human interpretation required</span>
      </div>
    </section>
    """, unsafe_allow_html=True)

    lr_path = MODEL_PATH
    rf_path = BASE_DIR / "models" / "random_forest.joblib"
    lr_model = joblib.load(lr_path) if lr_path.exists() else None
    rf_model = joblib.load(rf_path) if rf_path.exists() else None
    data_path = BASE_DIR / "data" / "StressLevelDataset.csv"
    dataset = None
    if data_path.exists():
        try:
            dataset = pd.read_csv(data_path)
        except Exception as exc:
            st.warning(f"Research dataset could not be loaded: {exc}")

    # Recorded results from the project's 80/20 stratified holdout evaluation.
    # Keep these clearly labelled as preliminary single-split results.
    metrics = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": 0.8818, "Precision": 0.8818, "Recall": 0.8821, "F1-score": 0.8820},
        {"Model": "Random Forest", "Accuracy": 0.8818, "Precision": 0.8826, "Recall": 0.8821, "F1-score": 0.8816},
    ])
    st.markdown("### 01 / Model readiness")
    a, b, c, d = st.columns(4)
    with a: st.markdown(f'<div class="metric"><div class="metric-label">MODEL ARTIFACTS</div><div class="metric-value">{sum(x is not None for x in [lr_model, rf_model])}/2</div><p>Available trained models</p></div>', unsafe_allow_html=True)
    with b: st.markdown(f'<div class="metric"><div class="metric-label">INPUT FEATURES</div><div class="metric-value">{len(FEATURES)}</div><p>Ordered model inputs</p></div>', unsafe_allow_html=True)
    with c: st.markdown(f'<div class="metric"><div class="metric-label">RESEARCH ROWS</div><div class="metric-value">{f"{len(dataset):,}" if dataset is not None else "—"}</div><p>Loaded secondary dataset</p></div>', unsafe_allow_html=True)
    with d: st.markdown('<div class="metric"><div class="metric-label">EVALUATION SPLIT</div><div class="metric-value">80 / 20</div><p>Stratified holdout experiment</p></div>', unsafe_allow_html=True)

    st.markdown("### 02 / Performance comparison")
    st.caption("Recorded evaluation metrics from the earlier single stratified holdout run. Values are not recalculated on this page.")
    metric_cols = st.columns(4)
    for col, metric_name in zip(metric_cols, ["Accuracy", "Precision", "Recall", "F1-score"]):
        with col:
            best = metrics.loc[metrics[metric_name].idxmax()]
            st.markdown(f'<div class="metric"><div class="metric-label">{metric_name.upper()}</div><div class="metric-value">{best[metric_name]*100:.2f}%</div><p>{best["Model"]}</p></div>', unsafe_allow_html=True)
    metric_long = metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
    perf_fig = px.bar(metric_long, x="Metric", y="Score", color="Model", barmode="group", text=metric_long["Score"].map(lambda x: f"{x*100:.2f}%"),
        title="Recorded holdout performance by model", color_discrete_sequence=["#13C8E8", "#FF4FA3"])
    perf_fig.update_layout(**PLOT_LAYOUT, height=370, yaxis_range=[0, 1.05], yaxis_title="Score", xaxis_title="")
    perf_fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(perf_fig, use_container_width=True, config=PLOT_CONFIG)
    display_metrics = metrics.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_metrics[col] = (display_metrics[col] * 100).round(2).astype(str) + "%"
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    st.markdown("### 03 / Global feature importance")
    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        importances = getattr(rf_model, "feature_importances_")
        if len(importances) == len(FEATURES):
            imp = pd.DataFrame({"Feature": FEATURES, "Importance": importances}).sort_values("Importance", ascending=False)
            top = imp.head(12).sort_values("Importance", ascending=True)
            imp_fig = px.bar(top, x="Importance", y="Feature", orientation="h", color="Importance",
                title="Random Forest · top 12 influential features",
                color_continuous_scale=[[0,"#164E63"],[.55,"#13C8E8"],[1,"#FF4FA3"]])
            imp_fig.update_layout(**PLOT_LAYOUT, height=470, coloraxis_showscale=False,
                xaxis_title="Relative importance", yaxis_title="")
            st.plotly_chart(imp_fig, use_container_width=True, config=PLOT_CONFIG)
            st.dataframe(imp.assign(Importance=imp["Importance"].round(4)), use_container_width=True, hide_index=True)
            st.caption("Global feature importance describes how the Random Forest used features across its fitted trees. It does not show causation or explain one individual prediction.")
        else:
            st.warning("The Random Forest feature count does not match the configured 20 input features. Check model and feature ordering.")
    else:
        st.warning("Random Forest feature importance is unavailable. Confirm models/random_forest.joblib exists and loads correctly.")

    st.markdown("### 04 / Research dataset composition")
    if dataset is not None and "stress_level" in dataset.columns:
        counts = dataset["stress_level"].value_counts().sort_index()
        class_frame = pd.DataFrame({"Dataset class": [f"Class {x}" for x in counts.index], "Records": counts.values})
        left, right = st.columns([1.1, 1])
        with left:
            class_fig = px.pie(class_frame, names="Dataset class", values="Records", hole=.62,
                title="Target class distribution", color_discrete_sequence=["#13C8E8", "#8B5CF6", "#FF4FA3"])
            class_fig.update_layout(**PLOT_LAYOUT, height=340)
            st.plotly_chart(class_fig, use_container_width=True, config=PLOT_CONFIG)
        with right:
            st.markdown('<div class="card"><h3>Dataset profile</h3><p><b>Rows:</b> '+f'{len(dataset):,}'+'</p><p><b>Columns:</b> '+f'{len(dataset.columns)}'+'</p><p><b>Target:</b> stress_level</p><p><b>Predictor features:</b> '+f'{len(FEATURES)}'+'</p><p><b>Missing cells:</b> '+f'{int(dataset.isna().sum().sum())}'+'</p></div>', unsafe_allow_html=True)
            st.caption("Class labels are shown exactly as encoded. Their semantic meaning has not been verified from a source codebook.")
    elif dataset is None:
        st.info("Dataset file not found at data/StressLevelDataset.csv. Add the project dataset to enable class composition charts.")

    st.markdown("### 05 / Model pipeline")
    pipeline_steps = ["20 structured inputs", "Feature-order alignment", "Logistic Regression + Random Forest", "Predicted dataset class + probability estimates", "Global feature explanation + what-if analysis"]
    st.markdown(" → ".join(f"**{step}**" for step in pipeline_steps))
    st.markdown('<div class="notice">Evaluation limitation: the displayed scores come from one 80/20 stratified holdout experiment. They are preliminary, may vary with a different split, and are not independent external validation. A confusion matrix is intentionally not displayed because the exact held-out predictions are not stored in the current app. Do not interpret dataset classes as clinical severity levels.</div>', unsafe_allow_html=True)

# ----------------------------- COUNSELOR STUDENT REVIEWS -----------------------------
elif st.session_state.page == "Student Reviews":
    st.markdown('<div class="kicker">COUNSELOR PORTAL  /  HUMAN-LED REVIEW</div>', unsafe_allow_html=True)
    st.title("Student Reviews")
    st.markdown("Review assessment submissions available in this active browser session. Records are labelled with anonymous session IDs; no student identity is collected by this prototype.")
    records = st.session_state.get("assessment_history", [])
    if "review_workflow" not in st.session_state:
        st.session_state.review_workflow = {}
    if not records:
        st.markdown("<div class='card'><h3>No assessment submissions yet</h3><p>There are currently no reviewable records in this session. Switch to the Student portal, complete a Wellbeing Assessment, and return here to review its model output.</p></div>", unsafe_allow_html=True)
        if st.button("Open Counselor Dashboard →"):
            st.session_state.page = "Counselor Dashboard"
            st.rerun()
    else:
        indexed = list(enumerate(records))[::-1]
        total = len(records)
        counts = {status: sum(1 for i, _ in indexed if st.session_state.review_workflow.get(i, {}).get("status", "Pending") == status) for status in ["Pending", "In Review", "Reviewed"]}
        cols = st.columns(4)
        for col, label, value in zip(cols, ["SESSION RECORDS", "PENDING", "IN REVIEW", "REVIEWED"], [total, counts["Pending"], counts["In Review"], counts["Reviewed"]]):
            with col:
                st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><p>Current browser session</p></div>', unsafe_allow_html=True)
        st.markdown("### Review queue")
        status_filter = st.selectbox("Filter records", ["All", "Pending", "In Review", "Reviewed"], key="review_filter")
        options, option_to_idx = [], {}
        for idx, rec in indexed:
            status = st.session_state.review_workflow.get(idx, {}).get("status", "Pending")
            if status_filter == "All" or status == status_filter:
                label = f"Session record {idx+1:03d} · {rec.get('timestamp', 'Time unavailable')} · Class {rec.get('class', '—')} · {status}"
                options.append(label)
                option_to_idx[label] = idx
        if not options:
            st.info("No records match this status filter.")
        else:
            selected = st.selectbox("Select an assessment record", options, key="selected_review_record")
            idx = option_to_idx[selected]
            record = records[idx]
            workflow = st.session_state.review_workflow.get(idx, {})
            left, right = st.columns([1, 1])
            with left:
                st.markdown(f'<div class="card"><div class="metric-label">ANONYMOUS SESSION RECORD</div><h2>#{idx+1:03d}</h2><p><b>Submitted:</b> {record.get("timestamp", "—")}</p><p><b>Predicted dataset class:</b> Class {record.get("class", "—")}</p><p><b>Review status:</b> {workflow.get("status", "Pending")}</p></div>', unsafe_allow_html=True)
            with right:
                probs = record.get("probabilities")
                if probs:
                    prob_df = pd.DataFrame({"Dataset class": [f"Class {i}" for i in range(len(probs))], "Model estimate (%)": [float(x)*100 for x in probs]})
                    fig = px.bar(prob_df, x="Dataset class", y="Model estimate (%)", color="Dataset class", title="Model probability estimates", color_discrete_sequence=["#13C8E8", "#8B5CF6", "#FF4FA3"])
                    fig.update_layout(**PLOT_LAYOUT, height=290, showlegend=False, yaxis_range=[0, 100])
                    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
                else:
                    st.info("Probability estimates were not stored for this record.")
            st.markdown("### Submitted indicator values")
            vals = record.get("values")
            if vals:
                view = pd.DataFrame([{"Indicator": k.replace("_", " ").title(), "Dataset-coded value": v} for k, v in vals.items()])
                st.dataframe(view, use_container_width=True, hide_index=True, height=420)
            else:
                st.warning("This older session record has no saved indicator values. New assessments will include them for review.")
            st.markdown("### Counselor review notes")
            statuses = ["Pending", "In Review", "Reviewed"]
            new_status = st.selectbox("Review status", statuses, index=statuses.index(workflow.get("status", "Pending")), key=f"review_status_{idx}")
            notes = st.text_area("Session note", value=workflow.get("notes", ""), height=120, key=f"review_notes_{idx}", placeholder="Record a factual observation or follow-up action. Avoid unnecessary identifying or sensitive details.")
            if st.button("Save review update", type="primary", use_container_width=True):
                st.session_state.review_workflow[idx] = {"status": new_status, "notes": notes}
                st.success("Review update saved for this active browser session.")
                st.rerun()
            st.caption("Review status and notes exist only in Streamlit session state. They are temporary, not encrypted case records, and may disappear when the session ends. Do not enter identifiable or highly sensitive information.")
    st.markdown('<div class="notice">Human-review safeguard: model outputs are predictions of research dataset labels, not diagnoses, verified wellbeing severity, or decisions about an individual. Use appropriate consent and secure institutional systems for any real-world deployment.</div>', unsafe_allow_html=True)

# ----------------------------- COUNSELOR DASHBOARD -----------------------------
elif st.session_state.page == "Counselor Dashboard":
    st.markdown('<div class="kicker">COUNSELOR PORTAL  /  OVERVIEW</div>', unsafe_allow_html=True)
    st.title("Counselor Dashboard")
    st.markdown("""
    <section class="assessment-hero">
      <div class="assessment-hero-inner">
        <div class="assessment-eyebrow">WELLQUEST AI · COUNSELOR WORKSPACE</div>
        <div class="assessment-heading">A clearer view of<br><span class="accent">student wellbeing patterns.</span></div>
        <div class="assessment-sub">Review the research dataset, inspect this session's assessment activity, and access model evaluation tools. Use every model output as a prompt for human review—not as a diagnosis or a decision about a student.</div>
        <span class="assessment-chip">✦ Research-backed prototype</span>
        <span class="assessment-chip">◈ Human-led review</span>
        <span class="assessment-chip">⌁ Transparent model outputs</span>
      </div>
    </section>
    """, unsafe_allow_html=True)

    history = st.session_state.get("assessment_history", [])
    latest = st.session_state.get("assessment_result")
    data_path = BASE_DIR / "data" / "StressLevelDataset.csv"
    dataset = None
    if data_path.exists():
        try:
            dataset = pd.read_csv(data_path)
        except Exception as exc:
            st.warning(f"Research dataset could not be read: {exc}")

    # These metrics deliberately distinguish local session activity from the secondary dataset.
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="metric"><div class="metric-label">SESSION CHECK-INS</div><div class="metric-value">{len(history)}</div><p>Assessments in this browser session</p></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric"><div class="metric-label">LATEST OUTPUT</div><div class="metric-value">{("Class " + str(latest["class"])) if latest else "—"}</div><p>Most recent model-predicted label</p></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric"><div class="metric-label">RESEARCH RECORDS</div><div class="metric-value">{f"{len(dataset):,}" if dataset is not None else "—"}</div><p>Rows in secondary research dataset</p></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric"><div class="metric-label">MODEL INPUTS</div><div class="metric-value">{len(FEATURES)}</div><p>Structured dataset indicators</p></div>', unsafe_allow_html=True)

    st.markdown("### Assessment activity")
    if history:
        activity = pd.DataFrame(history)
        activity["Check-in"] = range(1, len(activity) + 1)
        fig_activity = px.line(activity, x="Check-in", y="class", markers=True,
            title="Predicted dataset class across this session")
        fig_activity.update_traces(line=dict(color="#20C9EA", width=3),
            marker=dict(size=11, color="#FF4FA3", line=dict(color="#0B111B", width=2)))
        # Apply the shared layout first, then customise the y-axis separately.
        # This avoids passing the yaxis keyword twice when PLOT_LAYOUT already defines it.
        fig_activity.update_layout(**PLOT_LAYOUT, height=310,
            yaxis_title="Predicted dataset class", xaxis_title="Session check-in")
        fig_activity.update_yaxes(
            tickmode="array", tickvals=[0, 1, 2],
            ticktext=["Class 0", "Class 1", "Class 2"]
        )
        st.plotly_chart(fig_activity, use_container_width=True, config=PLOT_CONFIG)
        activity_view = activity[["timestamp", "class"]].rename(columns={"timestamp":"Check-in time", "class":"Predicted dataset class"})
        st.dataframe(activity_view.iloc[::-1], use_container_width=True, hide_index=True)
        st.caption("Session history is temporary and belongs to this browser session. It is not a permanent student record system.")
    else:
        st.markdown('<div class="card"><h3>No assessment activity yet</h3><p>Complete a Student Wellbeing Assessment in this session. The latest session activity will appear here automatically.</p></div>', unsafe_allow_html=True)

    st.markdown("### Research dataset overview")
    if dataset is not None and "stress_level" in dataset.columns:
        class_counts = dataset["stress_level"].value_counts().sort_index()
        left, right = st.columns([1.05, 1])
        with left:
            class_frame = pd.DataFrame({"Dataset class": [f"Class {x}" for x in class_counts.index], "Records": class_counts.values})
            fig_classes = px.bar(class_frame, x="Dataset class", y="Records", color="Dataset class",
                title="Research dataset class distribution",
                color_discrete_sequence=["#13C8E8", "#8B5CF6", "#FF4FA3"])
            fig_classes.update_layout(**PLOT_LAYOUT, height=330, showlegend=False)
            st.plotly_chart(fig_classes, use_container_width=True, config=PLOT_CONFIG)
        with right:
            st.markdown('<div class="card"><h3>Dataset facts</h3><p><b>Rows:</b> '+f'{len(dataset):,}'+'</p><p><b>Columns:</b> '+f'{len(dataset.columns)}'+'</p><p><b>Target field:</b> stress_level</p><p><b>Input indicators:</b> '+f'{len(FEATURES)}'+'</p></div>', unsafe_allow_html=True)
            st.caption("Class labels are shown as stored. Their wellbeing interpretations have not been verified from a source codebook.")
    elif dataset is None:
        st.info("The research dataset file was not found at data/StressLevelDataset.csv. The session activity section remains available.")
    else:
        st.warning("The dataset was loaded, but the expected stress_level target column was not found.")

    st.markdown("### Counselor tools")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="card"><h3>Model Analytics</h3><p>Inspect model artifacts, validation metrics, and global feature importance.</p></div>', unsafe_allow_html=True)
        if st.button("Open Model Analytics →", use_container_width=True):
            st.session_state.page = "Model Analytics"
            st.rerun()
    with c2:
        st.markdown('<div class="card"><h3>Student Reviews</h3><p>Open the review workspace for assessment records available in this session.</p></div>', unsafe_allow_html=True)
        if st.button("Open Student Reviews →", use_container_width=True):
            st.session_state.page = "Student Reviews"
            st.rerun()
    with c3:
        st.markdown('<div class="card"><h3>Reports</h3><p>Access research and assessment report exports as those workflows are completed.</p></div>', unsafe_allow_html=True)
        if st.button("Open Reports →", use_container_width=True):
            st.session_state.page = "Reports"
            st.rerun()

    st.markdown('<div class="notice">Privacy and interpretation: this prototype does not connect to an institutional student database, identify individual students, or persist records across sessions. Dataset class predictions are not diagnoses. Counselors should use appropriate consent, secure storage, and professional judgement in any real deployment.</div>', unsafe_allow_html=True)

elif st.session_state.page == "Reports":
    st.markdown('<div class="kicker">COUNSELOR PORTAL  /  DOCUMENT CENTRE</div>', unsafe_allow_html=True)
    st.title("Reports")
    st.markdown("""
    <section class="assessment-hero">
      <div class="assessment-hero-inner">
        <div class="assessment-eyebrow">WELLQUEST AI · REPORT STUDIO</div>
        <div class="assessment-heading">Turn session data into<br><span class="accent">clear, shareable reports.</span></div>
        <div class="assessment-sub">Export available assessment records and transparent model evaluation summaries. Reports contain anonymous session records only; this prototype is not connected to an institutional student database.</div>
        <span class="assessment-chip">CSV data export</span>
        <span class="assessment-chip">PDF summary</span>
        <span class="assessment-chip">Research transparency</span>
      </div>
    </section>
    """, unsafe_allow_html=True)

    records = st.session_state.get("assessment_history", [])
    st.markdown("### 01 / Report overview")
    c1, c2, c3, c4 = st.columns(4)
    summary_items = [("SESSION RECORDS", len(records)), ("MODEL INPUTS", len(FEATURES)),
                     ("AVAILABLE MODELS", sum((BASE_DIR / "models" / name).exists() for name in ["logistic_regression.joblib", "random_forest.joblib"])),
                     ("RESEARCH ROWS", "1,100")]
    for col, (label, value) in zip([c1,c2,c3,c4], summary_items):
        with col:
            st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><p>Current project information</p></div>', unsafe_allow_html=True)

    st.markdown("### 02 / Assessment report")
    if not records:
        st.info("No assessment records are available in this active session yet. Complete a Student Wellbeing Assessment to enable individual and session exports.")
    else:
        labels = [f"Session record {i+1:03d} · {r.get('timestamp','Time unavailable')} · Class {r.get('class','—')}" for i, r in enumerate(records)]
        chosen_label = st.selectbox("Choose an assessment", labels, key="report_selected_record")
        chosen_idx = labels.index(chosen_label)
        record = records[chosen_idx]
        probs = record.get("probabilities") or []
        report_rows = [{"Indicator": k, "Dataset-coded value": v} for k, v in (record.get("values") or {}).items()]
        assessment_df = pd.DataFrame(report_rows, columns=["Indicator", "Dataset-coded value"])
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f'<div class="card"><div class="metric-label">SELECTED ANONYMOUS RECORD</div><h2>#{chosen_idx+1:03d}</h2><p><b>Submitted:</b> {record.get("timestamp", "—")}</p><p><b>Predicted dataset class:</b> Class {record.get("class", "—")}</p><p><b>Stored indicators:</b> {len(report_rows)} / {len(FEATURES)}</p></div>', unsafe_allow_html=True)
        with right:
            if probs:
                prob_df = pd.DataFrame({"Dataset class": [f"Class {i}" for i in range(len(probs))], "Model estimate (%)": [round(float(x)*100, 3) for x in probs]})
                prob_fig = px.bar(prob_df, x="Dataset class", y="Model estimate (%)", color="Dataset class", title="Stored model probability estimates", color_discrete_sequence=["#13C8E8", "#8B5CF6", "#FF4FA3"])
                prob_fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False, yaxis_range=[0,100])
                st.plotly_chart(prob_fig, use_container_width=True, config=PLOT_CONFIG)
            else:
                st.info("No probability estimates were stored for this record.")
        st.dataframe(assessment_df, use_container_width=True, hide_index=True)
        csv_bytes = assessment_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download selected assessment CSV", data=csv_bytes,
            file_name=f"wellquest_assessment_{chosen_idx+1:03d}.csv", mime="text/csv", use_container_width=True)

        st.markdown("### 03 / Session export")
        flat_records = []
        for i, item in enumerate(records, start=1):
            row = {"session_record": f"{i:03d}", "timestamp": item.get("timestamp"), "predicted_dataset_class": item.get("class")}
            for feature in FEATURES:
                row[feature] = (item.get("values") or {}).get(feature)
            stored_probs = item.get("probabilities") or []
            for j, probability in enumerate(stored_probs):
                row[f"class_{j}_estimate"] = probability
            flat_records.append(row)
        session_df = pd.DataFrame(flat_records)
        st.download_button("Download all session assessments (CSV)", data=session_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="wellquest_session_assessments.csv", mime="text/csv", use_container_width=True)

    st.markdown("### 04 / Model evaluation report")
    metrics_df = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": 0.8818, "Precision": 0.8818, "Recall": 0.8821, "F1-score": 0.8820},
        {"Model": "Random Forest", "Accuracy": 0.8818, "Precision": 0.8826, "Recall": 0.8821, "F1-score": 0.8816},
    ])
    metrics_display = metrics_df.copy()
    for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
        metrics_display[metric] = (metrics_display[metric] * 100).round(2).astype(str) + "%"
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)
    st.caption("These are previously recorded preliminary results from one 80/20 stratified holdout evaluation; they are not recalculated during report export.")
    st.download_button("Download model metrics (CSV)", data=metrics_display.to_csv(index=False).encode("utf-8-sig"),
        file_name="wellquest_model_evaluation.csv", mime="text/csv", use_container_width=True)

    st.markdown("### 05 / PDF summary")
    st.write("Create a concise PDF containing the selected record (if available), stored indicator values, model evaluation metrics and research-use limitations.")
    if st.button("Generate PDF report", type="primary", use_container_width=True):
        try:
            from io import BytesIO
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name="WQTitle", parent=styles["Title"], textColor=colors.HexColor("#087FA5"), spaceAfter=8))
            story = [Paragraph("WELLQUEST AI", styles["WQTitle"]), Paragraph("Student Wellbeing Research & Assessment Summary", styles["Heading2"]),
                     Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}", styles["Normal"]), Spacer(1, 8*mm),
                     Paragraph("Assessment session overview", styles["Heading2"]),
                     Paragraph(f"Available session assessments: {len(records)} | Model inputs: {len(FEATURES)} | Research dataset: 1,100 rows", styles["Normal"]), Spacer(1, 5*mm)]
            if records:
                selected = records[chosen_idx]
                story += [Paragraph(f"Selected anonymous record #{chosen_idx+1:03d}", styles["Heading2"]),
                          Paragraph(f"Timestamp: {selected.get('timestamp','—')} | Predicted dataset class: Class {selected.get('class','—')}", styles["Normal"]), Spacer(1, 3*mm)]
                table_data = [["Indicator", "Dataset-coded value"]] + [[str(x["Indicator"]).replace("_", " ").title(), str(x["Dataset-coded value"])] for x in report_rows]
                if len(table_data) > 1:
                    tbl = Table(table_data, colWidths=[105*mm, 55*mm], repeatRows=1)
                    tbl.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#D9F2F8")),("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#073B4C")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#B8C7CE")),("PADDING",(0,0),(-1,-1),5)]))
                    story += [tbl, Spacer(1, 5*mm)]
                if selected.get("probabilities"):
                    story.append(Paragraph("Stored model estimates: " + " | ".join(f"Class {i}: {float(v)*100:.2f}%" for i,v in enumerate(selected["probabilities"])), styles["Normal"]))
            story += [Spacer(1, 5*mm), Paragraph("Recorded preliminary model evaluation", styles["Heading2"])]
            metric_table = [["Model", "Accuracy", "Precision", "Recall", "F1-score"]]
            for _, row in metrics_df.iterrows():
                metric_table.append([row["Model"]] + [f"{row[m]*100:.2f}%" for m in ["Accuracy","Precision","Recall","F1-score"]])
            mt = Table(metric_table, colWidths=[48*mm,27*mm,27*mm,27*mm,27*mm], repeatRows=1)
            mt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#D9F2F8")),("GRID",(0,0),(-1,-1),.35,colors.HexColor("#B8C7CE")),("PADDING",(0,0),(-1,-1),5)]))
            story += [mt, Spacer(1, 6*mm), Paragraph("Interpretation and limitations", styles["Heading2"]),
                      Paragraph("This academic prototype uses a secondary research dataset. Class labels are shown as encoded; their semantic meaning has not been verified from a source codebook. Model predictions are not diagnoses, clinical risk scores, causal explanations, or decisions about an individual. Evaluation figures are preliminary single-holdout results and do not establish real-world validity. Session records are temporary and are not permanent institutional records.", styles["Normal"])]
            doc.build(story)
            st.download_button("Download generated PDF report", data=buffer.getvalue(), file_name="WELLQUEST_AI_Report.pdf", mime="application/pdf", use_container_width=True)
            st.success("PDF report generated. Use the download button above to save it.")
        except ImportError:
            st.error("PDF export requires ReportLab. Install it in your active environment with: pip install reportlab")
        except Exception as exc:
            st.error(f"Could not generate the PDF report: {exc}")

    st.markdown('<div class="notice">Privacy and interpretation: exports contain only data available in this active Streamlit session and project-level research metrics. No student identity is collected. Do not add identifiable or highly sensitive information. Model outputs are not diagnoses.</div>', unsafe_allow_html=True)

# ----------------------------- OTHER MODULES -----------------------------
elif st.session_state.page == "Growth Journey":
    st.markdown('<div class="kicker">YOUR PROGRESS SPACE  /  REFLECT AND REVIEW</div>', unsafe_allow_html=True)
    st.title("Growth Journey")
    st.write("A personal space to build consistency, reflect on check-ins and celebrate small actions.")

    hist = st.session_state.get("assessment_history", [])
    result = st.session_state.get("assessment_result")
    journey_xp = 50 * len(hist)
    if "journey_completed_goals" not in st.session_state:
        st.session_state.journey_completed_goals = []
    if "journey_reflection" not in st.session_state:
        st.session_state.journey_reflection = ""

    # Progress cards: XP is earned for platform participation, never for a predicted class.
    level = 1 + journey_xp // 200
    level_xp = journey_xp % 200
    progress = level_xp / 200
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric"><div class="metric-label">JOURNEY XP</div><div class="metric-value">{journey_xp}</div><div class="small-note">50 XP per completed check-in</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric"><div class="metric-label">CURRENT LEVEL</div><div class="metric-value">{level}</div><div class="small-note">Consistency level</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric"><div class="metric-label">CHECK-INS</div><div class="metric-value">{len(hist)}</div><div class="small-note">This browser session</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric"><div class="metric-label">GOALS COMPLETED</div><div class="metric-value">{len(st.session_state.journey_completed_goals)}</div><div class="small-note">Personal action goals</div></div>', unsafe_allow_html=True)

    st.markdown("### Your next level")
    st.progress(progress, text=f"{level_xp} / 200 XP towards Level {level + 1}")
    st.caption("XP rewards participation and reflection only. It does not measure wellbeing or improvement.")

    st.markdown("### Achievement collection")
    achievements = [
        ("FIRST CHECK-IN", "Complete your first assessment", len(hist) >= 1),
        ("PATTERN EXPLORER", "Complete three assessments", len(hist) >= 3),
        ("CONSISTENCY BUILDER", "Complete five assessments", len(hist) >= 5),
        ("ACTION TAKER", "Complete one personal goal", len(st.session_state.journey_completed_goals) >= 1),
        ("GOAL COLLECTOR", "Complete three personal goals", len(st.session_state.journey_completed_goals) >= 3),
    ]
    ach_cols = st.columns(5)
    for i, (name, desc, unlocked) in enumerate(achievements):
        with ach_cols[i]:
            color = "#13C8E8" if unlocked else "#718096"
            st.markdown(f'''<div style="min-height:145px;padding:14px;border:1px solid {color}66;border-radius:15px;background:linear-gradient(145deg,rgba(19,200,232,.10),rgba(255,79,163,.04));text-align:center;">
            <div style="font-size:28px;color:{color};">{'✦' if unlocked else '◇'}</div>
            <div style="font-size:11px;font-weight:800;letter-spacing:1px;color:{color};margin:8px 0;">{name}</div>
            <div style="font-size:12px;color:#B8C7D2;">{desc}</div>
            <div style="font-size:10px;margin-top:8px;color:{color};">{'UNLOCKED' if unlocked else 'LOCKED'}</div></div>''', unsafe_allow_html=True)

    st.markdown("### Weekly action goals")
    st.write("Choose small actions that feel realistic. Tick them when you complete them; these are self-directed goals, not medical advice.")
    goal_options = ["Take a short screen break", "Plan one manageable study task", "Spend time on a relaxing activity", "Connect with a friend or trusted person", "Keep a consistent sleep routine tonight"]
    selected_goals = st.multiselect("Choose your goals", goal_options, default=[g for g in st.session_state.journey_completed_goals if g in goal_options], key="journey_goal_picker")
    st.session_state.journey_completed_goals = [g for g in st.session_state.journey_completed_goals if g in selected_goals]
    done_goals = []
    for goal in selected_goals:
        if st.checkbox(goal, value=goal in st.session_state.journey_completed_goals, key="goal_" + str(goal)):
            done_goals.append(goal)
    st.session_state.journey_completed_goals = done_goals
    st.caption(f"{len(done_goals)} of {len(selected_goals)} selected goals completed")

    st.markdown("### Check-in timeline")
    if not hist:
        st.info("Your timeline will appear after you complete a wellbeing assessment. No sample progress is generated.")
    else:
        journey = pd.DataFrame(hist).copy()
        journey["Check-in"] = range(1, len(journey) + 1)
        fig = px.scatter(journey, x="Check-in", y="class", color="class", hover_data=["timestamp"],
                         title="Recorded model-output history", color_continuous_scale=["#13C8E8", "#8B5CF6", "#FF4FA3"])
        fig.update_traces(marker=dict(size=15, line=dict(width=1, color="#EAF8FF")))
        fig.update_layout(**PLOT_LAYOUT, height=350, yaxis_title="Predicted dataset class", xaxis_title="Recorded check-in")
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)
        st.dataframe(journey[["timestamp", "class"]].rename(columns={"timestamp":"Check-in time", "class":"Predicted dataset class"}), use_container_width=True, hide_index=True)
        st.caption("This timeline displays stored dataset-class outputs only. Class changes are not evidence of recovery, deterioration or a validated wellbeing trend.")

    st.markdown("### Personal reflection")
    reflection = st.text_area("What is one thing you would like to focus on next?", value=st.session_state.journey_reflection, height=110, placeholder="Write a short note for yourself. Avoid entering identifying or highly sensitive information.")
    if st.button("Save my reflection", use_container_width=True):
        st.session_state.journey_reflection = reflection
        st.success("Reflection saved for this browser session.")
    st.markdown('<div class="notice">Privacy note: goals and reflections are stored only in the active Streamlit session and may disappear when the session ends. XP and badges are engagement features, not wellbeing scores. WELLQUEST AI is an academic prototype, not a diagnostic or emergency service.</div>', unsafe_allow_html=True)

elif st.session_state.page == "Support Hub":
    st.markdown('<div class="kicker">HUMAN-CENTRED SUPPORT  /  PRACTICAL TOOLS AND REAL HELP</div>', unsafe_allow_html=True)
    st.title("Support Hub")
    st.markdown("A calm space to find a next step, prepare a conversation, and connect with human support.")

    # Urgent help must be visible and must never depend on model predictions.
    st.markdown('''
    <div style="border:1px solid rgba(255,79,100,.55);border-radius:18px;padding:20px 22px;
                background:linear-gradient(110deg,rgba(255,79,100,.13),rgba(18,31,48,.92));margin:12px 0 22px;">
      <div style="font-size:12px;letter-spacing:2px;font-weight:800;color:#FF8792;">NEED URGENT HELP?</div>
      <div style="font-size:22px;font-weight:800;margin:8px 0;color:#F4F8FC;">Your safety comes first.</div>
      <div style="color:#C7D4E0;line-height:1.65;">If you or someone else is in immediate danger, call <b>112</b> in India or go to the nearest emergency department. Do not wait for an assessment or use this prototype to decide whether help is needed.</div>
      <div style="margin-top:12px;font-size:15px;font-weight:800;color:#FFFFFF;">India emergency helpline: 112</div>
    </div>''', unsafe_allow_html=True)

    st.markdown("### Choose the kind of support you need")
    support_paths = {
        "I feel overwhelmed": ("Start with one small step", ["Pause and take a few slow, comfortable breaths.", "Name one thing that needs attention today.", "Choose a task that can be done in 10–15 minutes.", "Tell someone you trust that things feel heavy."]),
        "Academic pressure": ("Make the workload manageable", ["Write down every pending task without sorting it yet.", "Mark the one task with the nearest deadline.", "Break that task into the smallest next action.", "Ask a faculty member or mentor about priorities if deadlines conflict."]),
        "Sleep or routine concerns": ("Reset one part of your routine", ["Choose a realistic time to begin winding down.", "Reduce bright screens shortly before sleep if practical.", "Keep tomorrow's first task simple and prepared.", "If sleep problems persist or affect daily life, speak with a qualified professional."]),
        "Relationship or social concerns": ("Reach out safely", ["Choose a person who usually listens without judgement.", "Describe what happened using one or two clear examples.", "Say what kind of help would be useful: listening, advice, or practical support.", "If you feel unsafe with someone, prioritise distance and seek trusted help."]),
        "I am not sure": ("You do not need to have the words yet", ["Take a moment somewhere you feel reasonably safe.", "Try writing: 'Right now, I am finding ___ difficult.'", "Consider talking to a trusted person even if the issue is unclear.", "Professional support can help you sort through uncertainty."]),
    }
    chosen_path = st.selectbox("What would you like help with right now?", list(support_paths.keys()), key="support_path_choice")
    path_title, path_steps = support_paths[chosen_path]
    st.markdown(f'<div class="card"><div class="kicker">A SUGGESTED NEXT STEP</div><h3>{path_title}</h3></div>', unsafe_allow_html=True)
    for i, step in enumerate(path_steps, 1):
        st.markdown(f"**{i}.** {step}")

    st.markdown("### A quick reset")
    st.write("Try this optional grounding exercise. Stop if it feels uncomfortable; choose another calming activity instead.")
    with st.expander("Open the 5–4–3–2–1 grounding exercise"):
        st.markdown("- **5** things you can see.\n- **4** things you can feel physically.\n- **3** things you can hear.\n- **2** things you can smell.\n- **1** thing you can taste, or one slow breath you can notice.")
        st.caption("This is a general grounding exercise, not a treatment or substitute for professional care.")

    st.markdown("### Prepare to talk to someone")
    st.write("You can create a short message to a trusted friend, family member, faculty mentor or campus support contact. Review it before sharing.")
    support_person = st.text_input("Who would you like to contact? (optional)", placeholder="e.g. friend, mentor, family member", key="support_person")
    message_topic = st.text_input("What has been difficult? (keep it general)", placeholder="e.g. study pressure, sleep, feeling overwhelmed", key="support_message_topic")
    message_help = st.selectbox("What would help most?", ["I would like someone to listen.", "I would like help planning a next step.", "I would like help finding professional support.", "I do not know yet; I just do not want to be alone."], key="support_message_help")
    if st.button("Prepare my message", use_container_width=True, key="support_prepare_message"):
        if not message_topic.strip():
            st.warning("Add a brief, general topic first, or use the suggested message below.")
        else:
            recipient = support_person.strip() or "Hi"
            st.session_state.support_prepared_message = f"Hi {recipient}, I have been finding {message_topic.strip()} difficult lately. {message_help} Would you be available to talk when you can?"
    if "support_prepared_message" in st.session_state:
        st.text_area("Your message (you can edit and copy it)", value=st.session_state.support_prepared_message, height=120, key="support_message_preview")
        st.caption("This message is only prepared in this app. WELLQUEST AI does not send it.")

    st.markdown("### Professional and emergency support in India")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown('''<div class="card" style="min-height:190px;"><div class="kicker">24 × 7 MENTAL HEALTH SUPPORT</div><h3>Tele-MANAS</h3><div style="font-size:27px;font-weight:850;color:#20C9EA;">14416</div><p>Free, confidential, round-the-clock tele-mental health support across India. You can also call 1800-891-4416.</p></div>''', unsafe_allow_html=True)
        st.markdown("[Official Tele-MANAS information](https://dghs.mohfw.gov.in/national-mental-health-programme.php)")
    with r2:
        st.markdown('''<div class="card" style="min-height:190px;"><div class="kicker">IMMEDIATE EMERGENCY</div><h3>Emergency Response</h3><div style="font-size:27px;font-weight:850;color:#FF8792;">112</div><p>India's integrated emergency response number for urgent police, fire and medical assistance.</p></div>''', unsafe_allow_html=True)
        st.markdown("[Official 112 emergency service](https://112.gov.in/)")
    st.markdown("[Student Support Resources – Government of India](https://ntf.education.gov.in/student-support-resources/)")

    st.markdown("### Save a personal next step")
    action_options = ["Speak with a trusted person", "Contact a faculty mentor", "Plan one manageable study task", "Take a short restorative break", "Explore professional support"]
    chosen_action = st.selectbox("Choose one action you may take next", action_options, key="support_action_choice")
    action_note = st.text_input("Optional reminder note", placeholder="e.g. after class tomorrow", key="support_action_note")
    if st.button("Save my next step", use_container_width=True, key="support_save_action"):
        st.session_state.support_saved_action = {"action": chosen_action, "note": action_note.strip(), "saved_at": datetime.now().strftime("%d %b %Y, %I:%M %p")}
        st.success("Your next step is saved for this Streamlit session.")
    saved_action = st.session_state.get("support_saved_action")
    if saved_action:
        st.markdown(f'<div class="card"><div class="kicker">YOUR SAVED NEXT STEP</div><h3>{saved_action["action"]}</h3><p>{saved_action["note"] or "No reminder note added."}</p><div class="small-note">Saved {saved_action["saved_at"]} · This browser session only</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="notice">Important: WELLQUEST AI is an academic screening-support prototype, not a diagnostic tool, counselling service or emergency service. Model predictions do not determine what support someone deserves. The support steps above are general information; seek qualified professional help for persistent or serious concerns. Do not enter names, contact details, or highly sensitive personal information here.</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="kicker">WELLQUEST MODULE</div>', unsafe_allow_html=True)
    st.title(st.session_state.page)
    st.info("This workspace is reserved for future institutional workflows. No live student records are connected in this prototype.")
    st.markdown('<div class="card"><h3>WELLQUEST AI</h3><p>Designed around transparent model outputs, thoughtful insights and human-centred support.</p></div>', unsafe_allow_html=True)
