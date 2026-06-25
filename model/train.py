"""
P2 · Lead Scoring — Training LightGBM v2
Near-real-time · Features GA4/VTEX + canal_origen · Sin CRM
Etiqueta: venta concretada en ≤7 días (del CRM, solo para entrenamiento)
"""
import sys, pandas as pd, numpy as np, lightgbm as lgb, shap, json, pickle, warnings, os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ks_2samp
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore'); np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

CONFIG = {
    "model_name": "lead_scoring_lgbm_v2",
    "version":    "v2",
    "target":     "convertido",
    "score_buckets": {"A": 0.75, "B": 0.50, "C": 0.30},
    "lgbm_params": {
        "objective":"binary","metric":"auc","boosting_type":"gbdt",
        "num_leaves":63,"learning_rate":0.04,"n_estimators":900,
        "feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,
        "min_child_samples":15,"reg_alpha":0.1,"reg_lambda":0.1,
        "random_state":42,"verbose":-1,"n_jobs":-1,
    }
}

# Features: GA4/VTEX + canal de entrada. Sin datos del CRM.
FEATURES_NUM = [
    'hora_lead','dia_semana','has_vtex',
    'tiempo_pagina_s','paginas_sesion','uso_comparador',
    'items_carrito','sesiones_72h',
    'is_prime_time','is_weekend',
    'is_begin_checkout','is_add_to_cart','is_sin_vtex',
    'is_mobile','is_paid','is_wsp_inbound','is_formulario',
    'tiempo_log','engagement_score','intent_score',
]
FEATURES_CAT = ['canal_origen','device','source_medium','producto_visto','cat_producto','punto_abandono']
FEATURES_ALL = FEATURES_NUM + FEATURES_CAT

print("="*60)
print("P2 · LEAD SCORING — TRAINING v2 (near-RT · GA4/VTEX + canal)")
print("="*60)
df = pd.read_csv(os.path.join(DATA_DIR, 'leads_data.csv'))
print(f"\n[DATA] {len(df):,} leads | Conversión: {df['convertido'].mean():.2%}")
print(f"       Con VTEX: {df['has_vtex'].mean():.1%} | Sin VTEX: {(1-df['has_vtex']).mean():.1%}")

# Feature engineering — todo derivable en tiempo real en Cloud Function
df['is_prime_time']      = df['hora_lead'].isin([18,19,20,21]).astype(int)
df['is_weekend']         = df['dia_semana'].isin([5,6]).astype(int)
df['is_begin_checkout']  = (df['punto_abandono']=='begin_checkout').astype(int)
df['is_add_to_cart']     = (df['punto_abandono']=='add_to_cart').astype(int)
df['is_sin_vtex']        = (df['punto_abandono']=='sin_sesion_vtex').astype(int)
df['is_mobile']          = (df['device']=='mobile').astype(int)
df['is_paid']            = df['source_medium'].isin(['google_cpc','facebook_ads']).astype(int)
df['is_wsp_inbound']     = (df['canal_origen']=='whatsapp_inbound').astype(int)
df['is_formulario']      = (df['canal_origen']=='formulario_web').astype(int)
df['tiempo_log']         = np.log1p(df['tiempo_pagina_s'])
df['engagement_score']   = (
    df['is_begin_checkout']*4
    + df['is_add_to_cart']*2
    + df['uso_comparador']*3
    + df['tiempo_log']*0.5
    + df['sesiones_72h']*0.8
    + df['paginas_sesion']*0.3
).round(4)
df['intent_score']       = (
    df['is_wsp_inbound']*3
    + df['is_formulario']*1.5
    + df['is_begin_checkout']*4
    + df['uso_comparador']*2
    + (df['sesiones_72h']>=2).astype(int)*2
).round(2)

le_dict = {}
for col in FEATURES_CAT:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

df = df.sort_values('fecha_lead').reset_index(drop=True)
n=len(df); n_test=int(n*.15); n_val=int(n*.15)
df_train=df.iloc[:n-n_test-n_val]; df_val=df.iloc[n-n_test-n_val:n-n_test]; df_test=df.iloc[n-n_test:]
X_train,y_train=df_train[FEATURES_ALL],df_train['convertido']
X_val,  y_val  =df_val[FEATURES_ALL],  df_val['convertido']
X_test, y_test =df_test[FEATURES_ALL], df_test['convertido']
print(f"\n[SPLIT] Train:{len(X_train):,} | Val:{len(X_val):,} | Test:{len(X_test):,}")

model = lgb.LGBMClassifier(**CONFIG["lgbm_params"])
model.fit(X_train,y_train,eval_set=[(X_val,y_val)],
          callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(200)])

prob = model.predict_proba(X_test)[:,1]
auc  = roc_auc_score(y_test,prob)
ap   = average_precision_score(y_test,prob)
ks,_ = ks_2samp(prob[y_test==1],prob[y_test==0])
n_top= int(len(y_test)*.20)
pat20= y_test.iloc[np.argsort(prob)[::-1][:n_top]].mean()
lift = pat20/y_test.mean()

print(f"\n{'─'*50}")
print(f"  AUC-ROC:           {auc:.4f}")
print(f"  Average Precision: {ap:.4f}")
print(f"  KS Statistic:      {ks:.4f}")
print(f"  Precision@top20%:  {pat20:.4f}")
print(f"  Lift@top20%:       {lift:.2f}x")
print(f"{'─'*50}")

b=CONFIG["score_buckets"]
TIER_MAP={"A · Urgente":{"sla":"< 1 hora","accion":"Llamada inmediata · agente senior"},
          "B · Alta":   {"sla":"< 4 horas","accion":"Cola prioritaria · oferta preparada"},
          "C · Media":  {"sla":"< 24 horas","accion":"Cola estándar · nurturing previo"},
          "D · Baja":   {"sla":"Sin contacto","accion":"Descarte temporal"}}
def tier(s):
    if s>=b['A']: return "A · Urgente"
    if s>=b['B']: return "B · Alta"
    if s>=b['C']: return "C · Media"
    return "D · Baja"

df_test=df_test.copy(); df_test['score']=prob; df_test['tier']=df_test['score'].apply(tier)
bstats=df_test.groupby('tier').agg(n=('convertido','count'),conv=('convertido','mean'),score=('score','mean')).round(3)
bstats['pct']=(bstats['n']/len(df_test)*100).round(1)
print("\n[TIERS]"); print(bstats.to_string())

print("\n[VTEX vs NO-VTEX] Conversión por disponibilidad de señal VTEX:")
vtx=df_test.groupby('has_vtex').agg(n=('convertido','count'),conv=('convertido','mean'),score=('score','mean')).round(3)
print(vtx.to_string())

print("\n[SHAP] Calculando...")
explainer=shap.TreeExplainer(model)
shap_s=X_test.sample(500,random_state=42)
shap_v=explainer.shap_values(shap_s)
if isinstance(shap_v,list): shap_v=shap_v[1]
shap_imp=pd.DataFrame({'feature':FEATURES_ALL,'shap_mean':np.abs(shap_v).mean(axis=0)}).sort_values('shap_mean',ascending=False)
print("\n  Top 10 features (GA4/VTEX + canal_origen):")
for _,r in shap_imp.head(10).iterrows():
    print(f"  {r['feature']:25s} {r['shap_mean']:.4f}  {'█'*int(r['shap_mean']*45)}")

os.makedirs(MODEL_DIR,exist_ok=True)
model.booster_.save_model(os.path.join(MODEL_DIR,'lgbm_leadscoring.txt'))
metadata={
    "model_name":CONFIG["model_name"],"version":CONFIG["version"],
    "train_date":str(pd.Timestamp.now().date()),
    "features":FEATURES_ALL,"features_num":FEATURES_NUM,"features_cat":FEATURES_CAT,
    "le_dict":{k:list(v.classes_) for k,v in le_dict.items()},
    "score_buckets":CONFIG["score_buckets"],"tier_map":TIER_MAP,
    "metrics":{"auc_roc":round(auc,4),"average_prec":round(ap,4),
               "ks_statistic":round(ks,4),"precision_top20":round(pat20,4),"lift_top20":round(lift,4)},
    "shap_top10":shap_imp.head(10)[['feature','shap_mean']].to_dict('records'),
    "best_iteration":int(model.best_iteration_),
    "arquitectura":"near_real_time · Pub/Sub → Cloud_Function → Vertex_AI_Endpoint → CRM",
    "features_eliminados_vs_v1":["plan_actual_crm","antigüedad_meses","nps_historico","visitas_30d","es_reactivo"],
    "nota":"CRM solo para etiqueta historica de entrenamiento. No usado en scoring RT.",
}
json.dump(metadata,open(os.path.join(MODEL_DIR,'metadata.json'),'w',encoding='utf-8'),indent=2,ensure_ascii=False)
pickle.dump(le_dict,open(os.path.join(MODEL_DIR,'label_encoders.pkl'),'wb'))
df_test[['lead_id','score','tier','convertido','canal_origen','has_vtex']].to_csv(
    os.path.join(MODEL_DIR,'test_scores.csv'),index=False)
print(f"\n[OK] Modelo v2 guardado | AUC-ROC: {auc:.4f}")
print(f"     Features: GA4/VTEX + canal_origen · Sin CRM")
