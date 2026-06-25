# P2 · Lead Scoring — Telco (v2)

## Arquitectura corregida
**Near-real-time** · Pub/Sub → Cloud Function → Vertex AI Endpoint → CRM
Sin datos del CRM para el scoring · Features GA4/VTEX + canal de origen

```
Lead capturado (formulario / WhatsApp / botón)
    ↓ webhook → Pub/Sub topic:leads-capturados
Cloud Function activada
    → consulta BQ: features sesión GA4 del client_id (si existe)
    → POST /predict a Vertex AI Endpoint
    ↓
score + tier + SHAP top-3 en < 300ms
    → escribe en tabla scores_leads (BQ)
    → notifica al CRM del call center vía API
Latencia total: < 5 minutos desde captura del lead
```

## Features del modelo (GA4/VTEX + canal — sin CRM)
- canal_origen (whatsapp_inbound / formulario_web / boton_ayuda) ← NUEVO vs v1
- hora_lead, dia_semana ← NUEVO vs v1
- has_vtex (1 si hay sesión VTEX linkeable)
- punto_abandono (begin_checkout / add_to_cart / view_item / sin_sesion_vtex)
- producto_visto, cat_producto
- uso_comparador, tiempo_pagina_s, paginas_sesion
- items_carrito, sesiones_72h, device, source_medium
- intent_score (compuesto: canal×3 + checkout×4 + comparador×2 + sesiones×2)
- engagement_score (compuesto: checkout×4 + comparador×3 + add_to_cart×2 + ...)

## Features eliminados vs v1
- plan_actual_crm, antigüedad_meses, nps_historico (CRM — no disponibles en RT)
- es_reactivo (redundante con canal_origen)
- visitas_30d (RFM — no aplica)

## Manejo de leads sin sesión VTEX (31% del dataset)
- WhatsApp directo sin visita previa a la tienda
- LightGBM maneja nulos nativamente
- El modelo usa canal_origen + hora como señal principal
- Score degradado pero funcional (Tier B en lugar de A)

## Tiers del call center
- A · Urgente (≥0.75): < 1 hora, agente senior
- B · Alta    (≥0.50): < 4 horas, oferta preparada
- C · Media   (≥0.30): < 24 horas, nurturing previo
- D · Baja    (<0.30): sin contacto, re-score si nueva actividad

## Costo GCP (producción)
~$139–$371/mes: BigQuery + Pub/Sub + Cloud Functions + Vertex AI + Monitoring + Looker

## Ejecución local

> Este proyecto corre junto a `claro_p1` en la misma máquina (otro par API + dashboard).
> Usa puertos explícitos para evitar que ambos Streamlit terminen apuntando al mismo backend.

```powershell
# Setup (una sola vez)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python model/train.py

# Terminal 1 — API (Vertex AI Endpoint simulado)
.\venv\Scripts\uvicorn.exe api.main:app --port 8002

# Terminal 2 — Dashboard
.\venv\Scripts\streamlit.exe run app\dashboard_p2.py --server.port 8502
```

| Servicio | Puerto |
|---|---|
| API (`api/main.py`) | 8002 |
| Dashboard (`app/dashboard_p2.py`) | 8502 |

Nota: `dashboard_p2.py` carga el modelo directamente desde `model/` (no llama a la API por HTTP);
el endpoint `8002` simula el flujo de producción (Cloud Function → Vertex AI) para pruebas con `curl`/Postman.

## Vistas del dashboard
1. Cola del call center (lista priorizada + VTEX vs sin VTEX)
2. Scoring de lead en vivo (payload JSON real → Vertex AI)
3. Análisis de features GA4 (canal, hora, punto abandono, VTEX disponible)
4. SHAP Explicabilidad (intent_score como feature dominante)
5. Simulador near-RT (flujo completo + comparativa lead con/sin VTEX + score decay)
