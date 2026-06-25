"""
P2 · Lead Scoring — Claro Perú × Attach Analytics · 2026
streamlit run p2_lead_scoring_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random, hashlib

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2 · Lead Scoring | Claro",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── ESTILOS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #EBF3FA; }
  [data-testid="stSidebar"] { background: #1B3A6B; }
  [data-testid="stSidebar"] * { color: #fff !important; }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label { color: rgba(255,255,255,.75) !important; }
  h1 { color: #1B3A6B !important; }
  h2 { color: #2E5FA3 !important; border-bottom: 2px solid #D5E4F5; padding-bottom: 6px; }
  h3 { color: #1B3A6B !important; }
  .tier-hot    { background:#FDECEA; border-left:4px solid #7A1A1A; border-radius:8px; padding:14px 16px; }
  .tier-warm   { background:#FFF3CD; border-left:4px solid #EF9F27; border-radius:8px; padding:14px 16px; }
  .tier-cold   { background:#D5E4F5; border-left:4px solid #1B3A6B; border-radius:8px; padding:14px 16px; }
  .tier-nurture{ background:#F2F4F7; border-left:4px solid #888;    border-radius:8px; padding:14px 16px; }
  .kpi-box     { background:#fff; border:1px solid #E0E0E0; border-radius:10px; padding:14px 18px; text-align:center; }
  .score-bar   { background:#D5E4F5; border-radius:20px; height:10px; margin:4px 0; }
  .pipeline-step { background:#fff; border:1px solid #D5E4F5; border-radius:8px; padding:10px; text-align:center; font-size:12px; }
  .badge-hot   { background:#7A1A1A; color:#fff; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
  .badge-warm  { background:#EF9F27; color:#fff; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
  .badge-cold  { background:#2E5FA3; color:#fff; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
  .badge-nurture{ background:#888;   color:#fff; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTES ────────────────────────────────────────────────────────────
PLANES   = ["Plan 50GB", "Plan 100GB", "Plan Ilimitado", "Equipo + Plan"]
DEVICES  = ["Mobile", "Desktop", "Tablet"]
HORAS    = list(range(0, 24))
DIAS     = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

TIER_CONFIG = {
    "Hot":    {"color":"#7A1A1A", "bg":"#FDECEA", "emoji":"🔴", "sla":"< 1 hora",
               "canal":"Llamada directa", "canal_icon":"📞", "css":"tier-hot",   "badge":"badge-hot"},
    "Warm":   {"color":"#633806", "bg":"#FFF3CD", "emoji":"🟡", "sla":"< 4 horas",
               "canal":"Llamada o WhatsApp", "canal_icon":"📞💬", "css":"tier-warm", "badge":"badge-warm"},
    "Cold":   {"color":"#1B3A6B", "bg":"#D5E4F5", "emoji":"🔵", "sla":"< 24 horas",
               "canal":"Llamada programada", "canal_icon":"📅", "css":"tier-cold",  "badge":"badge-cold"},
    "Nurture":{"color":"#5A5A5A", "bg":"#F2F4F7", "emoji":"⚫", "sla":"Flujo auto",
               "canal":"Email automatizado", "canal_icon":"📧", "css":"tier-nurture","badge":"badge-nurture"},
}

# ── FUNCIONES DE SCORING ──────────────────────────────────────────────────
def calc_engagement_score(begin_checkout, uso_comparador, add_to_cart, tiempo_pagina_min, sesiones_72h):
    """Fórmula central del modelo P2."""
    return (begin_checkout * 4 +
            uso_comparador  * 3 +
            add_to_cart     * 2 +
            np.log1p(tiempo_pagina_min) * 0.5 +
            sesiones_72h    * 0.8)

def calc_propension(engagement_score, hora, dia_semana, plan, device, punto_abandono):
    """Score de propensión de conversión (0–1)."""
    score = 0.20

    # Engagement score normalizado (max ~20)
    score += min(engagement_score / 20, 0.35)

    # Punto de abandono
    if punto_abandono == "begin_checkout":  score += 0.18
    elif punto_abandono == "add_to_cart":   score += 0.10
    elif punto_abandono == "view_item":     score += 0.04

    # Hora
    if 18 <= hora <= 21:   score += 0.08
    elif 9 <= hora <= 17:  score += 0.04

    # Día semana
    if dia_semana in [0,1,2,3,4]:  score += 0.04  # Lun-Vie

    # Plan
    if "Ilimitado" in plan:   score += 0.06
    elif "100GB" in plan:     score += 0.04
    elif "Equipo" in plan:    score += 0.03

    # Device
    if device == "Mobile" and 18 <= hora <= 21:  score += 0.05

    noise = np.random.normal(0, 0.02)
    return float(np.clip(score + noise, 0.05, 0.97))

def assign_tier(score):
    if score >= 0.72:  return "Hot"
    elif score >= 0.48: return "Warm"
    elif score >= 0.28: return "Cold"
    else:               return "Nurture"

def generar_guion_llm(tier, plan, hora, device, engagement_score, punto_abandono):
    """Simula el output del guión LLM dinámico."""
    hora_label = f"{hora}:00h"
    urgencia = {
        "Hot":    "Este lead tiene alta intención de compra. Oferta directa.",
        "Warm":   "Lead con interés activo. Resolver dudas y cerrar.",
        "Cold":   "Lead con intención media. Información + oferta.",
        "Nurture":"Lead sin señal fuerte. Presentar opciones básicas.",
    }[tier]

    punto_label = {
        "begin_checkout": "llegó hasta el proceso de contratación",
        "add_to_cart":    "agregó el plan al carrito",
        "view_item":      "revisó los detalles del plan",
        "sin_sesion":     "sin historial digital previo",
    }.get(punto_abandono, "navegó el sitio")

    apertura = {
        "Hot":  f"Hola [nombre], le llamo de Claro sobre su interés en el {plan} que revisó hoy. "
                f"Vi que {punto_label} — ¿tiene unos minutos para completar la contratación?",
        "Warm": f"Hola [nombre], le contactamos de Claro. Notamos su interés en el {plan}. "
                f"¿Hay alguna duda que podamos resolver para ayudarle a decidir?",
        "Cold": f"Hola [nombre], soy de Claro. Revisó nuestro {plan}. "
                f"¿Le puedo dar más información sobre los beneficios?",
        "Nurture": f"Hola [nombre], le contactamos de Claro para contarle sobre nuestras "
                   f"ofertas actuales en planes móviles.",
    }[tier]

    return {
        "contexto_agente": urgencia,
        "apertura":        apertura,
        "engagement":      f"Engagement score: {engagement_score:.2f} | Punto: {punto_label}",
        "timing":          f"Lead recibido: {hora_label} | Device: {device}",
        "plan_interes":    plan,
    }

# ── DATOS SINTÉTICOS PARA COLA ────────────────────────────────────────────
@st.cache_data
def generar_cola_leads(n=30, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    planes_pool   = PLANES * 8
    nombres_pool  = ["García M.","López R.","Martínez K.","Rodríguez A.","Sánchez J.",
                     "Torres L.","Flores D.","Rivera C.","Gomez P.","Díaz S.",
                     "Morales F.","Reyes N.","Cruz O.","Herrera B.","Mendoza T."]
    puntos_pool   = ["begin_checkout","add_to_cart","view_item","begin_checkout","add_to_cart"]
    devices_pool  = ["Mobile","Mobile","Desktop","Mobile","Tablet"]
    horas_pool    = [18,19,20,21,14,15,16,10,11,9,22,17,13]

    leads = []
    for i in range(n):
        bc  = random.choice([0,0,1,1,1])
        uc  = random.choice([0,0,1,1])
        atc = random.choice([0,1,1,1])
        tp  = round(random.uniform(0.5, 8.0), 1)
        s72 = random.randint(1, 5)
        eng = calc_engagement_score(bc, uc, atc, tp, s72)
        hora     = random.choice(horas_pool)
        dia      = random.randint(0, 6)
        plan     = random.choice(planes_pool)
        device   = random.choice(devices_pool)
        punto    = random.choice(puntos_pool)
        score    = calc_propension(eng, hora, dia, plan, device, punto)
        tier     = assign_tier(score)

        leads.append({
            "lead_id":       f"L{1000+i}",
            "nombre":        random.choice(nombres_pool),
            "plan":          plan,
            "punto_abandono":punto,
            "device":        device,
            "hora":          hora,
            "dia":           DIAS[dia],
            "engagement":    round(eng, 2),
            "score":         round(score, 3),
            "tier":          tier,
            "canal":         TIER_CONFIG[tier]["canal"],
            "sla":           TIER_CONFIG[tier]["sla"],
            "emoji":         TIER_CONFIG[tier]["emoji"],
            "hace_min":      random.randint(2, 90),
        })

    df = pd.DataFrame(leads)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["posicion"] = df.index + 1
    return df

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 P2 · Lead Scoring")
    st.markdown("**Claro Perú × Attach Analytics**")
    st.markdown("---")
    pagina = st.radio("Navegación", [
        "🎯 Simulador",
        "📞 Cola Call Center",
        "📊 Modelo BQML",
        "🏗️ Arquitectura",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("BigQuery ML · Firestore · Cloud Run · Dataform")
    st.markdown("---")
    st.caption("Canal de entrada: **Formulario web**")
    st.caption("Clave JOIN: email / tel. hasheado ↔ GA4")
    st.caption("Fase 0B: validación antes de F1–F6")


# ══════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — SIMULADOR
# ══════════════════════════════════════════════════════════════════════════
if pagina == "🎯 Simulador":
    st.title("🎯 Simulador · Lead Scoring P2")
    st.markdown("Simula el hot path completo: **formulario web → JOIN GA4 → engagement score → BQML → tier → canal → guión LLM**")

    col_form, col_ga4 = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("📋 Datos del formulario web")
        plan = st.selectbox("Plan de interés", PLANES)
        col_h, col_d = st.columns(2)
        with col_h:
            hora = st.selectbox("Hora de envío", HORAS, index=18,
                                format_func=lambda h: f"{h}:00h")
        with col_d:
            dia_idx = st.selectbox("Día", range(7), format_func=lambda i: DIAS[i], index=0)
        device = st.selectbox("Device", DEVICES)

        st.markdown("---")
        st.markdown("**🔗 JOIN formulario ↔ GA4**")
        email_input = st.text_input("Email (se hashea antes del JOIN)", "cliente@ejemplo.com")
        join_key = hashlib.sha256(email_input.encode()).hexdigest()[:16]
        st.caption(f"Clave JOIN: `{join_key}...` (SHA-256 truncado)")

    with col_ga4:
        st.subheader("📊 Historial GA4/BQ (enriquecimiento)")
        punto_abandono = st.selectbox("Punto de abandono en Vtex", [
            "begin_checkout", "add_to_cart", "view_item", "sin_sesion"
        ], format_func=lambda x: {
            "begin_checkout": "begin_checkout — señal muy fuerte",
            "add_to_cart":    "add_to_cart — señal fuerte",
            "view_item":      "view_item — señal media",
            "sin_sesion":     "Sin sesión previa en GA4",
        }[x])
        begin_checkout = 1 if punto_abandono == "begin_checkout" else 0
        add_to_cart    = 1 if punto_abandono in ["begin_checkout","add_to_cart"] else 0

        uso_comparador = st.toggle("Usó el comparador de planes", value=True)
        sesiones_72h   = st.slider("Sesiones en las últimas 72h", 1, 8, 2)
        tiempo_pagina  = st.slider("Tiempo en página del plan (minutos)", 0.5, 10.0, 3.0, 0.5)

    # ── CÁLCULO ──────────────────────────────────────────────────────────
    eng  = calc_engagement_score(begin_checkout, int(uso_comparador), add_to_cart,
                                  tiempo_pagina, sesiones_72h)
    prop = calc_propension(eng, hora, dia_idx, plan, device, punto_abandono)
    tier = assign_tier(prop)
    cfg  = TIER_CONFIG[tier]
    guion = generar_guion_llm(tier, plan, hora, device, eng, punto_abandono)

    st.markdown("---")
    st.subheader("⚡ Resultado · Hot path (&lt;500ms)")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Engagement Score", f"{eng:.2f}")
    k2.metric("Score propensión", f"{prop:.3f}")
    k3.metric("Tier asignado", f"{cfg['emoji']} {tier}")
    k4.metric("SLA canal", cfg["sla"])

    # Tier card
    st.markdown(f"""
    <div class="{cfg['css']}">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <div style="font-size:18px;font-weight:700;color:{cfg['color']}">{cfg['emoji']} Tier {tier} — {cfg['canal']} {cfg['canal_icon']}</div>
          <div style="font-size:13px;color:#666;margin-top:4px">SLA: {cfg['sla']} · {plan} · {DIAS[dia_idx]} {hora}:00h · {device}</div>
        </div>
        <div style="font-size:32px">{cfg['canal_icon']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Desglose engagement score
    col_eng, col_guion = st.columns([1, 1], gap="large")

    with col_eng:
        st.subheader("📐 Desglose Engagement Score")
        componentes = {
            "begin_checkout ×4": begin_checkout * 4,
            "comparador ×3":     int(uso_comparador) * 3,
            "add_to_cart ×2":    add_to_cart * 2,
            "tiempo_log ×0.5":   round(np.log1p(tiempo_pagina) * 0.5, 3),
            "sesiones ×0.8":     sesiones_72h * 0.8,
        }
        fig = go.Figure(go.Bar(
            y=list(componentes.keys()),
            x=list(componentes.values()),
            orientation='h',
            marker_color=["#7A1A1A","#EF9F27","#534AB7","#2E5FA3","#2E5FA3"],
            text=[f"{v:.2f}" for v in componentes.values()],
            textposition='outside',
        ))
        fig.update_layout(
            height=220, margin=dict(l=0,r=40,t=10,b=10),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(gridcolor='#EBF3FA', range=[0, max(componentes.values())*1.3]),
            yaxis=dict(tickfont=dict(size=11)),
            font=dict(family="Segoe UI"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Total engagement score: `{eng:.2f}`**")

    with col_guion:
        st.subheader("🤖 Guión LLM · Agente call center")
        st.markdown(f"**Contexto:** _{guion['contexto_agente']}_")
        st.markdown("**Apertura sugerida:**")
        st.info(guion["apertura"])
        st.caption(f"📊 {guion['engagement']}")
        st.caption(f"⏱ {guion['timing']} | 📱 Plan: {guion['plan_interes']}")

    # Timeline
    st.subheader("⏱ Timeline hot path")
    steps = [
        ("📋 Formulario web", "t = 0ms"),
        ("🔗 JOIN GA4/BQ", "+5ms"),
        ("☁️ Cloud Run", "+15ms"),
        ("🗄 Firestore read", "+55ms"),
        ("⚖️ Filtros LGPD", "+70ms"),
        (f"{cfg['canal_icon']} Canal: {tier}", "~120ms"),
        ("📝 BQ log", "async"),
    ]
    cols = st.columns(len(steps))
    for col, (label, time) in zip(cols, steps):
        col.markdown(f"""
        <div class="pipeline-step">
          <div style="font-size:14px">{label}</div>
          <div style="font-size:10px;color:#EF9F27;font-weight:700;margin-top:4px">{time}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — COLA CALL CENTER
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "📞 Cola Call Center":
    st.title("📞 Cola Call Center — Leads priorizados por BQML")

    df = generar_cola_leads(30)

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total leads", len(df))
    k2.metric("🔴 Hot",    int((df.tier=="Hot").sum()),    delta="< 1h")
    k3.metric("🟡 Warm",   int((df.tier=="Warm").sum()),   delta="< 4h")
    k4.metric("🔵 Cold",   int((df.tier=="Cold").sum()),   delta="< 24h")
    k5.metric("⚫ Nurture", int((df.tier=="Nurture").sum()),delta="auto")

    # Filtro tier
    filtro = st.multiselect("Filtrar por tier", ["Hot","Warm","Cold","Nurture"],
                            default=["Hot","Warm","Cold"])
    df_vis = df[df.tier.isin(filtro)].copy()

    # Cola tabla
    st.subheader(f"Lista de leads · {len(df_vis)} mostrando")

    for _, row in df_vis.iterrows():
        cfg = TIER_CONFIG[row.tier]
        with st.container():
            c1, c2, c3, c4, c5, c6 = st.columns([0.5, 2.5, 2, 1.5, 1.5, 1.5])
            c1.markdown(f"<div style='font-size:22px;text-align:center'>{cfg['emoji']}</div>",
                        unsafe_allow_html=True)
            c2.markdown(f"**{row.nombre}** · `{row.lead_id}`  \n"
                        f"<span style='font-size:11px;color:#666'>{row.plan}</span>",
                        unsafe_allow_html=True)
            c3.markdown(f"<span style='font-size:11px'>📍 {row.punto_abandono}  \n"
                        f"📱 {row.device} · {row.dia} {row.hora}:00h</span>",
                        unsafe_allow_html=True)
            c4.metric("Score", f"{row.score:.3f}", label_visibility="collapsed")
            c5.markdown(f"<span class='{cfg['badge']}'>{row.tier}</span><br>"
                        f"<span style='font-size:10px;color:#666'>{cfg['sla']}</span>",
                        unsafe_allow_html=True)
            c6.markdown(f"{cfg['canal_icon']} {cfg['canal']}  \n"
                        f"<span style='font-size:10px;color:#888'>hace {row.hace_min} min</span>",
                        unsafe_allow_html=True)
            st.markdown("<hr style='margin:4px 0;border-color:#EBF3FA'>", unsafe_allow_html=True)

    # Distribución
    st.subheader("Distribución de scores")
    col_hist, col_pie = st.columns(2)

    with col_hist:
        fig_hist = go.Figure()
        color_map = {"Hot":"#7A1A1A","Warm":"#EF9F27","Cold":"#2E5FA3","Nurture":"#888"}
        for t in ["Nurture","Cold","Warm","Hot"]:
            sub = df[df.tier == t]
            fig_hist.add_trace(go.Histogram(
                x=sub.score, name=t, nbinsx=10,
                marker_color=color_map[t], opacity=0.8,
            ))
        fig_hist.update_layout(
            barmode='stack', height=280, margin=dict(l=0,r=0,t=10,b=10),
            plot_bgcolor='white', paper_bgcolor='white',
            legend=dict(orientation='h', y=1.15),
            xaxis_title="Score propensión", yaxis_title="Leads",
            font=dict(family="Segoe UI"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_pie:
        tier_counts = df.tier.value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=tier_counts.index,
            values=tier_counts.values,
            marker_colors=[color_map[t] for t in tier_counts.index],
            hole=0.45,
            textinfo='label+percent',
        ))
        fig_pie.update_layout(
            height=280, margin=dict(l=0,r=0,t=10,b=10),
            font=dict(family="Segoe UI"),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — MODELO BQML
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Modelo BQML":
    st.title("📊 Modelo BQML — Propensión de conversión")
    st.markdown("**BOOSTED_TREE_CLASSIFIER** entrenado sobre leads históricos con etiqueta CRM (venta ≤ 7 días). Canal de entrada único: **formulario web**.")

    # Feature importance simulada
    features = {
        "engagement_score":      0.28,
        "begin_checkout":        0.18,
        "uso_comparador_planes": 0.12,
        "sesiones_72h":          0.10,
        "tiempo_pagina_min":     0.09,
        "add_to_cart":           0.08,
        "hora_lead":             0.06,
        "plan_visto":            0.05,
        "dia_semana":            0.03,
        "device":                0.01,
    }

    col_fi, col_info = st.columns([1.4, 1], gap="large")

    with col_fi:
        st.subheader("Feature importance · ML.EXPLAIN_PREDICT")

        # Nota eliminación canal_origen_lead
        st.info("⚠️ `canal_origen_lead` eliminado — único canal de entrada (formulario web). "
                "Sin varianza → sin poder predictivo.")

        fig_fi = go.Figure(go.Bar(
            y=list(features.keys()),
            x=list(features.values()),
            orientation='h',
            marker_color=["#1B3A6B" if i==0 else
                          "#7A1A1A" if i==1 else
                          "#EF9F27" if i==2 else
                          "#534AB7" if i==3 else
                          "#2E5FA3"
                          for i in range(len(features))],
            text=[f"{v:.0%}" for v in features.values()],
            textposition='outside',
        ))
        fig_fi.update_layout(
            height=320, margin=dict(l=0,r=60,t=10,b=10),
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(tickformat='.0%', range=[0, 0.35], gridcolor='#EBF3FA'),
            yaxis=dict(tickfont=dict(size=11)),
            font=dict(family="Segoe UI"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_info:
        st.subheader("Configuración del modelo")
        st.markdown("""
        ```sql
        CREATE OR REPLACE MODEL
          claro.bqml_lead_scoring
        OPTIONS (
          model_type = 'BOOSTED_TREE_CLASSIFIER',
          input_label_cols = ['venta_7d'],
          num_parallel_tree = 6,
          max_iterations = 80,
          auto_class_weights = TRUE,
          data_split_method = 'RANDOM',
          data_split_eval_fraction = 0.2
        ) AS
        SELECT * FROM
          claro.feat_leads_formulario
        WHERE fecha_lead >= '2024-01-01'
        ```
        """)

        st.subheader("Métricas esperadas (Fase 0B)")
        m1, m2 = st.columns(2)
        m1.metric("AUC-ROC objetivo", "≥ 0.68")
        m2.metric("Precision Hot tier", "≥ 65%")
        m1.metric("Lift top 20%", "≥ 2.0×")
        m2.metric("Recall Hot", "≥ 55%")

    # Thresholds
    st.subheader("Thresholds de tier")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    for col, (tier, rng, desc) in zip(
        [col_t1, col_t2, col_t3, col_t4],
        [("🔴 Hot",    "≥ 0.72", "begin_checkout + comparador activo"),
         ("🟡 Warm",   "0.48–0.71", "add_to_cart + sesiones recientes"),
         ("🔵 Cold",   "0.28–0.47", "view_item + señal débil"),
         ("⚫ Nurture","< 0.28", "sin historial GA4 previo")]
    ):
        col.markdown(f"""
        <div class="kpi-box">
          <div style="font-size:14px;font-weight:700">{tier}</div>
          <div style="font-size:18px;font-weight:700;color:#1B3A6B;margin:6px 0">{rng}</div>
          <div style="font-size:10px;color:#888">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # JOIN key explanation
    st.subheader("JOIN Formulario ↔ GA4")
    st.markdown("""
    | Campo formulario | Campo GA4/BQ | Tipo hash | Cobertura esperada |
    |---|---|---|---|
    | `email` | `user_properties.email_hash` | SHA-256 | ~60–70% de leads |
    | `telefono` | `user_properties.tel_hash` | SHA-256 | ~40–55% de leads |
    | Fallback | Sin enriquecimiento GA4 | — | Tier Nurture automático |

    > ⚠️ **Objetivo Fase 0B:** validar cobertura real del JOIN antes de comprometer Fases 1–6.
    > Si cobertura < 40%, rediseñar arquitectura de captura del identificador.
    """)


# ══════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — ARQUITECTURA
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "🏗️ Arquitectura":
    st.title("🏗️ Arquitectura · Stack BigQuery ML")

    # Hot path
    st.subheader("⚡ Hot path · < 500ms")
    steps_hot = [
        ("📋", "Formulario web", "claro.com · email/tel hasheado"),
        ("☁️", "Cloud Run", "función determinística"),
        ("🗄", "Firestore", "score precomputado · <50ms"),
        ("⚖️", "Filtros", "LGPD · horario · freq cap"),
        ("📊", "Tier + Canal", "Hot/Warm/Cold/Nurture"),
        ("📞", "CRM", "cola priorizada · guión"),
        ("📝", "BQ Log", "feedback loop · async"),
    ]
    cols = st.columns(len(steps_hot))
    for col, (icon, title, sub) in zip(cols, steps_hot):
        col.markdown(f"""
        <div style="background:#fff;border:1px solid #D5E4F5;border-radius:8px;
                    padding:12px 8px;text-align:center;min-height:90px">
          <div style="font-size:20px">{icon}</div>
          <div style="font-size:11px;font-weight:700;color:#1B3A6B;margin:4px 0">{title}</div>
          <div style="font-size:9px;color:#888;line-height:1.4">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Batch path
    st.subheader("🔄 Batch path · quincenal")
    steps_batch = [
        ("📋", "Formulario web", "histórico leads"),
        ("🔗", "JOIN GA4/BQ", "email/tel hash ↔ eventos"),
        ("🔧", "Dataform", "feat_leads_formulario · SQL versionado"),
        ("🤖", "BQML", "BOOSTED_TREE · ML.PREDICT"),
        ("📊", "scores_leads", "score × tier × explain_top3"),
        ("🗄", "Firestore", "export batch · TTL 48h"),
    ]
    cols = st.columns(len(steps_batch))
    for col, (icon, title, sub) in zip(cols, steps_batch):
        col.markdown(f"""
        <div style="background:#EEEDFE;border:1px solid #534AB7;border-radius:8px;
                    padding:12px 8px;text-align:center;min-height:90px">
          <div style="font-size:20px">{icon}</div>
          <div style="font-size:11px;font-weight:700;color:#3C3489;margin:4px 0">{title}</div>
          <div style="font-size:9px;color:#534AB7;line-height:1.4">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Decisiones de diseño
    st.subheader("Decisiones de diseño")
    c1, c2 = st.columns(2)
    with c1:
        st.success("✅ **Email/tel hasheado como clave JOIN** — identidad determinística sin depender de cookies ni GA4 client_id probabilístico.")
        st.success("✅ **Firestore precomputado** — el submit del formulario dispara Cloud Run; el score se sirve en <50ms sin latencia adicional al usuario.")
        st.success("✅ **BQML batch** — sin infraestructura ML adicional. BOOSTED_TREE sobre el BigQuery que Claro ya opera.")
    with c2:
        st.success("✅ **Función determinística en Cloud Run** — ranking(score) → filtros(LGPD, horario, freq_caps) → canal óptimo. Sin LLM en el hot path.")
        st.success("✅ **Dataform SQL versionado** — linaje de datos, pruebas de calidad integradas, reentrenamiento quincenal.")
        st.success("✅ **CRM write-back de tiers** — el agente ve el tier, el plan de interés y el contexto GA4 antes de llamar.")

    # Fase 0B warning
    st.warning("""
    ⚠️ **Fase 0B obligatoria antes de F1–F6**

    La viabilidad del modelo depende de la cobertura del JOIN formulario ↔ GA4.
    La Fase 0B valida esta clave antes de comprometer las 155h del proyecto formal.

    **Gate 2 — criterios go/no-go:**
    - Cobertura JOIN ≥ 40% de leads históricos
    - AUC-ROC modelo base ≥ 0.65 sobre datos reales de Claro
    - Engagement score construible con los datos disponibles
    """)