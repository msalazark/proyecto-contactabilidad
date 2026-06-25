"""
P2 · Lead Scoring — API FastAPI v2
Near-real-time: simula el Vertex AI Endpoint invocado por Cloud Function
tras captura del lead vía Pub/Sub.
Producción: Vertex AI Endpoint, NO Cloud Run.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import lightgbm as lgb
import numpy as np, pandas as pd, shap, json, pickle, os
from datetime import datetime, timedelta

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH    = os.path.join(BASE_DIR, 'model', 'lgbm_leadscoring.txt')
METADATA_PATH = os.path.join(BASE_DIR, 'model', 'metadata.json')
LE_PATH       = os.path.join(BASE_DIR, 'model', 'label_encoders.pkl')

model    = lgb.Booster(model_file=MODEL_PATH)
metadata = json.load(open(METADATA_PATH, encoding='utf-8'))
le_dict  = pickle.load(open(LE_PATH,'rb'))
FEATURES_ALL = metadata['features']
BUCKETS      = metadata['score_buckets']
TIER_MAP     = metadata['tier_map']
explainer    = shap.TreeExplainer(model)

app = FastAPI(
    title="P2 · Lead Scoring — Vertex AI Endpoint (local)",
    description="""
    Scoring near-real-time de leads para priorización del call center de Claro.
    
    Flujo de producción:
    Lead capturado → Pub/Sub topic:leads-capturados → Cloud Function →
    (1) consulta BQ features sesión GA4 del client_id →
    (2) POST /predict a este Vertex AI Endpoint →
    (3) escribe score en tabla scores_leads BQ →
    (4) notifica CRM vía API
    
    Features: GA4/VTEX + canal_origen. SIN datos del CRM de Claro.
    Etiqueta de entrenamiento: venta en ≤7 días (solo del CRM histórico).
    """,
    version="2.0.0",
)

class LeadFeatures(BaseModel):
    lead_id:         str
    client_id:       str   = Field(..., description="GA4 client_id del lead")
    canal_origen:    str   = Field(..., description="whatsapp_inbound | formulario_web | boton_ayuda")
    hora_lead:       int   = Field(..., ge=0, le=23)
    dia_semana:      int   = Field(..., ge=0, le=6)
    has_vtex:        int   = Field(..., ge=0, le=1, description="1 si hay sesión VTEX asociable en 72h")
    producto_visto:  str   = Field(default='sin_dato')
    cat_producto:    str   = Field(default='sin_dato')
    punto_abandono:  str   = Field(default='sin_sesion_vtex')
    tiempo_pagina_s: int   = Field(default=0, ge=0)
    paginas_sesion:  int   = Field(default=0, ge=0)
    uso_comparador:  int   = Field(default=0, ge=0, le=1)
    items_carrito:   int   = Field(default=0, ge=0)
    sesiones_72h:    int   = Field(default=0, ge=0)
    device:          str   = Field(default='mobile')
    source_medium:   str   = Field(default='directo')

class ScoreResponse(BaseModel):
    lead_id:          str
    client_id:        str
    score:            float
    tier:             str
    sla_contacto:     str
    accion:           str
    shap_top3:        List[dict]
    score_expires_at: str
    scored_at:        str
    model_version:    str
    has_vtex_signal:  bool

def build_features(lead: LeadFeatures) -> pd.DataFrame:
    d = lead.model_dump()
    row = {
        'hora_lead':         d['hora_lead'],
        'dia_semana':        d['dia_semana'],
        'has_vtex':          d['has_vtex'],
        'tiempo_pagina_s':   d['tiempo_pagina_s'],
        'paginas_sesion':    d['paginas_sesion'],
        'uso_comparador':    d['uso_comparador'],
        'items_carrito':     d['items_carrito'],
        'sesiones_72h':      d['sesiones_72h'],
        'is_prime_time':     int(d['hora_lead'] in [18,19,20,21]),
        'is_weekend':        int(d['dia_semana'] in [5,6]),
        'is_begin_checkout': int(d['punto_abandono']=='begin_checkout'),
        'is_add_to_cart':    int(d['punto_abandono']=='add_to_cart'),
        'is_sin_vtex':       int(d['punto_abandono']=='sin_sesion_vtex'),
        'is_mobile':         int(d['device']=='mobile'),
        'is_paid':           int(d['source_medium'] in ['google_cpc','facebook_ads']),
        'is_wsp_inbound':    int(d['canal_origen']=='whatsapp_inbound'),
        'is_formulario':     int(d['canal_origen']=='formulario_web'),
        'tiempo_log':        np.log1p(d['tiempo_pagina_s']),
        'engagement_score':  round(
            int(d['punto_abandono']=='begin_checkout')*4
            +int(d['punto_abandono']=='add_to_cart')*2
            +d['uso_comparador']*3
            +np.log1p(d['tiempo_pagina_s'])*0.5
            +d['sesiones_72h']*0.8
            +d['paginas_sesion']*0.3, 4),
        'intent_score': round(
            int(d['canal_origen']=='whatsapp_inbound')*3
            +int(d['canal_origen']=='formulario_web')*1.5
            +int(d['punto_abandono']=='begin_checkout')*4
            +d['uso_comparador']*2
            +int(d['sesiones_72h']>=2)*2, 2),
        'canal_origen':   d['canal_origen'],
        'device':         d['device'],
        'source_medium':  d['source_medium'],
        'producto_visto': d['producto_visto'],
        'cat_producto':   d['cat_producto'],
        'punto_abandono': d['punto_abandono'],
    }
    df = pd.DataFrame([row])
    for col in ['canal_origen','device','source_medium','producto_visto','cat_producto','punto_abandono']:
        le = le_dict[col]; known = list(le.classes_)
        df[col] = df[col].apply(lambda x: x if x in known else known[0])
        df[col] = le.transform(df[col])
    return df[FEATURES_ALL]

def assign_tier(score: float) -> str:
    b = BUCKETS
    if score >= b['A']: return "A · Urgente"
    if score >= b['B']: return "B · Alta"
    if score >= b['C']: return "C · Media"
    return "D · Baja"

def get_shap_top3(df_feat):
    sv = explainer.shap_values(df_feat)
    if isinstance(sv,list): sv=sv[1]
    sv=sv[0]
    pairs=sorted(zip(FEATURES_ALL,sv),key=lambda x:abs(x[1]),reverse=True)[:3]
    return [{"feature":f,"shap_value":round(v,4),"direction":"↑" if v>0 else "↓"} for f,v in pairs]

def score_expiry():
    return (datetime.utcnow()+timedelta(hours=48)).isoformat()

@app.get("/health")
def health():
    return {"status":"ok","model":metadata['model_name'],"version":metadata['version'],
            "arquitectura":metadata['arquitectura'],"auc_roc":metadata['metrics']['auc_roc'],
            "nota":metadata['nota']}

@app.post("/predict", response_model=ScoreResponse)
def predict(lead: LeadFeatures):
    """
    Endpoint principal. En producción es invocado por Cloud Function
    tras recibir el evento del lead desde Pub/Sub.
    """
    try:
        df_feat = build_features(lead)
        score   = float(model.predict(df_feat)[0])
        tier    = assign_tier(score)
        info    = TIER_MAP.get(tier, {"sla":"—","accion":"—"})
        return ScoreResponse(
            lead_id=lead.lead_id, client_id=lead.client_id,
            score=round(score,4), tier=tier,
            sla_contacto=info['sla'], accion=info['accion'],
            shap_top3=get_shap_top3(df_feat),
            score_expires_at=score_expiry(),
            scored_at=datetime.utcnow().isoformat(),
            model_version=metadata['model_name'],
            has_vtex_signal=bool(lead.has_vtex),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(leads: List[LeadFeatures]):
    results=[]; dist={"A · Urgente":0,"B · Alta":0,"C · Media":0,"D · Baja":0}
    for lead in leads:
        try:
            df_feat=build_features(lead); score=float(model.predict(df_feat)[0])
            tier=assign_tier(score); info=TIER_MAP.get(tier,{"sla":"—","accion":"—"})
            dist[tier]+=1
            results.append({"lead_id":lead.lead_id,"score":round(score,4),"tier":tier,
                             "sla":info['sla'],"shap_top3":get_shap_top3(df_feat),
                             "has_vtex":lead.has_vtex,"scored_at":datetime.utcnow().isoformat()})
        except Exception as e:
            results.append({"lead_id":lead.lead_id,"error":str(e)})
    return {"total":len(results),"distribution":dist,"results":results,
            "scored_at":datetime.utcnow().isoformat()}

@app.get("/model/info")
def model_info():
    return metadata
