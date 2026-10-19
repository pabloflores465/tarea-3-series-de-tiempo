"""Ejecutar: python codigo/analisis.py. Rehace tablas, figuras y cifras del informe."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.diagnostic import acorr_ljungbox
from limpieza import limpiar
from estadistica import acf_manual, ljung_box, simular_ar1, resumir_replicas
from modelos import (indicadoras, ajustar_sarima, origen_movil,
                     cv_aleatoria, errores, fecha_rezago)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'resultados'
FIG = OUT / 'figuras'
DIAS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True,
                     'grid.alpha': .18, 'axes.prop_cycle': plt.cycler(color=[
                         '#2563a6', '#db7a27', '#27846c', '#ad4279', '#7764ae', '#555555'])})


def guardar(df, nombre, index=False):
    df.to_csv(OUT / (nombre + '.csv'), index=index, float_format='%.10g')


def figura(nombre):
    plt.tight_layout()
    plt.savefig(FIG / (nombre + '.png'), bbox_inches='tight')
    plt.close()


def dibujar_acf(ax, y, titulo):
    r = acf_manual(y, 21)
    ax.vlines(np.arange(1, 22), 0, r, color='#2563a6')
    ax.scatter(np.arange(1, 22), r, s=12)
    b = 1.96 / np.sqrt(len(y))
    ax.axhline(b, ls='--', color='#ad4279')
    ax.axhline(-b, ls='--', color='#ad4279')
    ax.axhline(0, color='#555555', lw=.7)
    ax.set(title=titulo, xlabel='Rezago (filas de días abiertos)', ylabel='ACF')
    return r


def parte0(resumen):
    filas = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for i, phi in enumerate([0, .5, .9, 1]):
        seed = 202600 + i
        y = simular_ar1(phi, replicas=5000, semilla=seed)
        filas.append(resumir_replicas(y, f'AR({phi:g})', seed, phi))
        axes.flat[i].plot(np.arange(1, 301), y[0], lw=.9)
        axes.flat[i].set(title=f'φ = {phi:g}', xlabel='Paso', ylabel='Valor')
    figura('01_procesos_ar')
    t = np.arange(1, 301)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    for i, (nombre, media) in enumerate([
        ('Tendencia', 2 * t / 300),
        ('Patrón semanal', np.where(t % 7 == 6, -1.5, .25))]):
        seed = 202604 + i
        y = media + np.random.default_rng(seed).normal(size=(5000, 300))
        filas.append(resumir_replicas(y, nombre, seed))
        axes[i].plot(t, y[0], lw=.8, label='Realización')
        axes[i].plot(t, media, lw=1.2, color='#db7a27', label='Media conocida')
        axes[i].set(title=nombre, xlabel='Paso', ylabel='Valor')
        axes[i].legend(loc='upper left', ncol=2)
    figura('02_medias_cambiantes')
    mc = pd.DataFrame(filas)
    guardar(mc, '00_simulacion')
    # Verificación independiente de las dos funciones implementadas a mano.
    ejemplo = simular_ar1(.5, semilla=42)[0]
    for m in [6, 10, 12]:
        propio = ljung_box(ejemplo, m)
        referencia = acorr_ljungbox(ejemplo, lags=[m], return_df=True).iloc[0]
        np.testing.assert_allclose([propio['Q'], propio['p']], referencia, rtol=1e-10)
    resumen['replicas'] = 5000


FERIADOS = {
    '2010-12-25': 'Navidad (también sábado)',
    '2010-12-26': 'Boxing Day',
    '2010-12-27': 'Navidad, feriado sustituto',
    '2010-12-28': 'Boxing Day, feriado sustituto',
    '2011-01-01': 'Año Nuevo (también sábado)',
    '2011-01-03': 'Año Nuevo, feriado sustituto',
    '2011-04-22': 'Viernes Santo',
    '2011-04-24': 'Domingo de Pascua: cierre compatible con Semana Santa',
    '2011-04-25': 'Lunes de Pascua',
    '2011-04-29': 'Boda real: feriado extraordinario',
    '2011-05-02': 'Early May Bank Holiday',
    '2011-05-30': 'Spring Bank Holiday',
    '2011-08-29': 'Summer Bank Holiday'}


def motivo_cierre(d):
    clave = d.strftime('%Y-%m-%d')
    if clave in FERIADOS:
        return FERIADOS[clave]
    if d.weekday() == 5:
        return 'Sábado: cierre semanal'
    if pd.Timestamp('2010-12-24') <= d <= pd.Timestamp('2011-01-04'):
        return 'Pausa de Navidad/Año Nuevo (inferencia por ausencia)'
    return 'Sin transacciones: causa no documentada'


def partes1y2(raw, resumen):
    resumen['original'] = {'filas': len(raw), 'facturas': raw.InvoiceNo.nunique(),
                           'productos': raw.StockCode.nunique(),
                           'clientes': raw.CustomerID.nunique(),
                           'sin_cliente': int(raw.CustomerID.isna().sum()),
                           'fraccion_sin_cliente': float(raw.CustomerID.isna().mean())}
    limpio = limpiar(raw)
    guardar(limpio.attrs['bitacora'], '01_bitacora')
    resumen['limpio'] = {'filas': len(limpio), 'monto': float(limpio.monto.sum()),
                         'sin_cliente': int(limpio.CustomerID.isna().sum()),
                         'devoluciones': int(limpio.InvoiceNo.str.startswith('C').sum())}
    especiales = raw.loc[~raw.StockCode.str.match(r'^\d{5}', na=False)]
    guardar(especiales.groupby(['StockCode', 'Description'], dropna=False)
            .agg(filas=('Quantity', 'size'), cantidad=('Quantity', 'sum'),
                 precio_min=('UnitPrice', 'min'), precio_max=('UnitPrice', 'max'))
            .reset_index(), '01_codigos_especiales')
    prefijos = raw.InvoiceNo.str.extract(r'^(\D+)')[0].fillna('Dígito')
    guardar(prefijos.value_counts().rename_axis('prefijo').reset_index(name='filas'),
            '01_prefijos')
    negativas = raw.loc[raw.Quantity.lt(0) & ~raw.InvoiceNo.str.startswith('C')]
    guardar(negativas.groupby('Description', dropna=False).size().reset_index(name='filas')
            .sort_values('filas', ascending=False), '01_ajustes_inventario')
    extremos = raw.assign(monto=raw.Quantity * raw.UnitPrice)
    top = extremos.loc[extremos.monto.abs().nlargest(6).index].copy()
    top['se_conserva'] = top.index.isin(limpio.index)
    guardar(top, '01_seis_extremos')
    guardar(extremos.loc[extremos.CustomerID.isin(top.CustomerID.unique())]
            .sort_values('InvoiceDate'), '01_historia_clientes_extremos')
    relacionados = []
    for idx, r in top.iterrows():
        pares = extremos.loc[extremos.StockCode.eq(r.StockCode) &
                             extremos.Quantity.eq(-r.Quantity) &
                             extremos.UnitPrice.eq(r.UnitPrice)].copy()
        pares['fila_extrema'] = idx
        relacionados.append(pares)
    guardar(pd.concat(relacionados, ignore_index=True), '01_contrapartidas_extremos')
    fechas_raw = pd.DatetimeIndex(raw.InvoiceDate.dt.normalize().unique()).sort_values()
    calendario = pd.date_range(fechas_raw.min(), fechas_raw.max(), freq='D')
    diario = limpio.groupby('fecha').monto.sum().reindex(calendario)
    diario.index.name = 'fecha'
    guardar(diario.rename('venta_gbp').reset_index(), '02_serie_diaria_completa')
    faltantes = calendario.difference(fechas_raw)
    guardar(pd.DataFrame({'fecha': faltantes,
                          'dia': [DIAS[d.weekday()] for d in faltantes],
                          'clasificacion': [motivo_cierre(d) for d in faltantes]}),
            '02_dias_sin_transacciones')
    ultimo = raw.loc[raw.InvoiceDate.dt.normalize().eq(calendario.max()), 'InvoiceDate']
    resumen['calendario'] = {'dias': len(calendario), 'dias_transacciones_raw': len(fechas_raw),
                             'dias_transacciones_limpio': int(diario.notna().sum()),
                             'dias_sin_transacciones': len(faltantes),
                             'ultimo_inicio': str(ultimo.min()), 'ultimo_fin': str(ultimo.max())}
    # El último día es parcial. Los cierres siguen siendo NaN.
    serie = diario.iloc[:-1]
    assert serie.dropna().gt(0).all(), 'Venta no positiva: revisar antes del log.'
    return limpio, serie


def partes3y4(serie, resumen, registro):
    train = serie.loc[:'2011-09-30']
    abiertos = train.dropna()
    ma = serie.fillna(0).rolling(7, center=True, min_periods=7).mean()
    ma_train = train.fillna(0).rolling(7, center=True, min_periods=7).mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(serie.index, serie, lw=.7, alpha=.65, label='Venta neta: día abierto')
    ax.plot(ma.index, ma, lw=1.8, label='Media centrada de 7 días, cierres = 0')
    ax.axvline(pd.Timestamp('2011-10-01'), color='#ad4279', ls='--', label='Corte de entrenamiento')
    ax.set(ylabel='GBP', xlabel='Fecha', title='Ventas y tendencia semanal')
    ax.legend(fontsize=8)
    figura('03_serie_tendencia')
    cociente = (train / ma_train).dropna()
    est = cociente.groupby(cociente.index.weekday).agg(['mean', 'count'])
    est['dia'] = [DIAS[d] for d in est.index]
    est.columns = ['indice', 'n', 'dia']
    guardar(est.reset_index(names='weekday'), '03_indices_estacionales')
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(est.dia, est.indice)
    ax.axhline(1, color='#ad4279', ls='--')
    ax.set(ylabel='Venta / media de 7 días', title='Índices semanales: solo entrenamiento')
    figura('04_indices_estacionales')
    dummies = indicadoras(abiertos.index)
    beta = np.linalg.lstsq(dummies, abiertos.to_numpy(), rcond=None)[0]
    e = abiertos.to_numpy() - dummies @ beta
    xx = np.column_stack([np.ones(len(e) - 1), e[:-1]])
    c, phi = np.linalg.lstsq(xx, e[1:], rcond=None)[0]
    u = e[1:] - c - phi * e[:-1]
    resumen['ar_residuos'] = {'c': float(c), 'phi': float(phi)}
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    acfs = {}
    for ax, nombre, valores in zip(axes, ['Venta original', 'Residuo de indicadoras',
                                           'Residuo de AR(1)'], [abiertos, e, u]):
        acfs[nombre] = dibujar_acf(ax, valores, nombre)
    figura('05_acf_original_y_residuos')
    guardar(pd.DataFrame({'lag': np.arange(1, 22), **acfs}), '03_04_acf')
    resumen['acf_original'] = {'n': len(abiertos),
                               'fuera': (np.flatnonzero(np.abs(acfs['Venta original']) >
                                                          1.96/np.sqrt(len(abiertos)))+1).tolist(),
                               'pico': int(acfs['Venta original'].argmax()+1)}
    diagnosticos = []
    for nombre, valores, k in [('Original', abiertos, 0), ('Indicadoras', e, 0), ('AR1', u, 1)]:
        for m in [6, 12, max(1, int(np.log(len(valores))))]:
            diagnosticos.append({'modelo': nombre, 'n': len(valores), 'm': m,
                                 **ljung_box(valores, m, k)})
    fits = {}
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.7))
    params = []
    for ax, nombre in zip(axes, ['S1', 'S2']):
        fit = ajustar_sarima(abiertos.to_numpy(), nombre, registro, 'ajuste_inicial')
        fits[nombre] = fit
        residuo = fit.resid[7:] if nombre == 'S2' else fit.resid
        dibujar_acf(ax, residuo, nombre + ': residuos en logaritmos')
        for m in [6, 12, max(3, int(np.log(len(residuo))))]:
            diagnosticos.append({'modelo': nombre, 'n': len(residuo), 'm': m,
                                 **ljung_box(residuo, m, 2)})
        params += [{'modelo': nombre, 'parametro': k, 'valor': float(v)}
                   for k, v in zip(fit.param_names, fit.params)]
    figura('06_acf_sarima')
    guardar(pd.DataFrame(params), '04_parametros_sarima')
    guardar(pd.DataFrame(diagnosticos), '04_ljung_box')
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for nombre, fit in fits.items():
        ax.plot(np.arange(1, 181), np.exp(fit.forecast(180)), label=nombre)
    ax.set(xlabel='Horizonte: días abiertos', ylabel='GBP', title='Pronóstico largo desde el cierre del entrenamiento')
    ax.legend()
    figura('07_horizonte_largo')
    resumen['medias'] = {'entrenamiento': float(abiertos.mean()),
                         'prueba': float(serie.loc['2011-10-03':'2011-12-08'].mean())}
    # Esta variable usa el futuro dentro del entrenamiento: solo para 5.4.
    return ma_train


def parte5(serie, ma_train, resumen, registro):
    cv = cv_aleatoria(serie, ma_train)
    guardar(cv, '05_cv_por_semilla')
    valid, fv = origen_movil(serie, ma_train, 'validacion', registro)
    prueba, fp = origen_movil(serie, ma_train, 'prueba', registro)
    pred = pd.concat([valid, prueba], ignore_index=True)
    guardar(pred, '05_predicciones')
    guardar(pd.concat([fv, fp], ignore_index=True), '05_fuentes_rezagos')
    filas = []
    for modelo in ['B1', 'B2', 'M1', 'M2', 'S1', 'S2', 'TRAMPA']:
        row = {'modelo': modelo}
        g = cv.loc[cv.modelo.eq(modelo)]
        for medida in ['MAE', 'RMSE']:
            row['a_' + medida] = g[medida].mean() if len(g) else np.nan
            row['a_' + medida + '_sd'] = g[medida].std(ddof=1) if len(g) else np.nan
        row['a_n'] = int(g.n.iloc[0]) if len(g) else 0
        for clave, datos in [('b', valid), ('c', prueba)]:
            g = datos.loc[datos.modelo.eq(modelo)]
            row[clave + '_n'] = len(g)
            for medida in ['MAE', 'RMSE']:
                row[clave + '_' + medida] = errores(g.real, g.pred)[medida] if len(g) else np.nan
        filas.append(row)
    tabla = pd.DataFrame(filas)
    guardar(tabla, '05_comparacion_modelos')
    mejor_mae = tabla.loc[tabla.c_MAE.idxmin(), 'modelo']
    mejor_rmse = tabla.loc[tabla.c_RMSE.idxmin(), 'modelo']
    resumen['ganadores'] = {'MAE': mejor_mae, 'RMSE': mejor_rmse}
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=True)
    for ax, nombre in zip(axes.flat, ['B1', 'B2', 'M1', 'M2', 'S1', 'S2']):
        p = prueba.loc[prueba.modelo.eq(nombre)]
        ax.plot(p.fecha, p.real, color='#555555', lw=1, label='Real')
        ax.plot(p.fecha, p.pred, color='#db7a27', lw=1, label=nombre)
        ax.set(title=nombre, ylabel='GBP')
        ax.tick_params(axis='x', rotation=25)
        ax.legend(fontsize=8)
    figura('08_pronosticos_prueba')
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sub = tabla.loc[tabla.modelo.isin(['M2', 'TRAMPA'])]
    pos = np.arange(2)
    ax.bar(pos-.16, sub.a_MAE, width=.32, label='CV aleatoria')
    ax.bar(pos+.16, sub.b_MAE, width=.32, label='Origen móvil')
    ax.set(xticks=pos, xticklabels=sub.modelo, ylabel='MAE (GBP)', title='La fuga también engaña al origen móvil')
    ax.legend()
    figura('09_trampa')


def parte6(limpio, resumen):
    conocidos = limpio.loc[limpio.CustomerID.notna()].copy()
    corte = pd.Timestamp('2011-10-01')
    antes = conocidos.loc[conocidos.InvoiceDate < corte]
    ventas = antes.loc[~antes.InvoiceNo.str.startswith('C') & antes.Quantity.gt(0)]
    rfm = antes.groupby('CustomerID').monto.sum().rename('M').to_frame()
    ultima = ventas.groupby('CustomerID').fecha.max()
    rfm['R'] = (corte - ultima).dt.days.reindex(rfm.index)
    rfm['sin_compra_previa'] = rfm.R.isna()
    # Solo devoluciones: R no existe. Se ubica detrás de todos los compradores.
    rfm['R'] = rfm.R.fillna((corte - antes.fecha.min()).days + 1).astype(int)
    rfm['F'] = ventas.groupby('CustomerID').InvoiceNo.nunique().reindex(rfm.index).fillna(0).astype(int)
    rfm = rfm.sort_index()
    rfm = rfm.iloc[np.random.default_rng(2026).permutation(len(rfm))].copy()
    rfm['puntaje_R'] = pd.qcut(-rfm.R, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['puntaje_F'] = pd.qcut(rfm.F.rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['puntaje_M'] = pd.qcut(rfm.M, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['suma'] = rfm[['puntaje_R', 'puntaje_F', 'puntaje_M']].sum(axis=1)
    despues = conocidos.loc[conocidos.fecha.between('2011-10-01', '2011-12-08')]
    compras_despues = despues.loc[~despues.InvoiceNo.str.startswith('C') & despues.Quantity.gt(0)]
    rfm['compra'] = rfm.index.isin(compras_despues.CustomerID.unique()).astype(int)
    rfm['campeon'] = rfm[['puntaje_R', 'puntaje_F', 'puntaje_M']].ge(4).all(axis=1)
    guardar(rfm.sort_index().reset_index(), '06_clientes_rfm')
    conversion, aucs = [], []
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, dimension in zip(axes, ['R', 'F', 'M']):
        g = rfm.groupby('puntaje_' + dimension).compra.agg(['mean', 'count', 'sum'])
        for puntaje, row in g.iterrows():
            conversion.append({'dimension': dimension, 'puntaje': puntaje,
                               'clientes': int(row['count']), 'compraron': int(row['sum']),
                               'fraccion_compra': float(row['mean'])})
        ax.bar(g.index, g['mean'])
        ax.set(title=dimension, xlabel='Puntaje (5 = mejor)', ylim=(0, 1), xticks=range(1, 6))
    axes[0].set_ylabel('Fracción que volvió a comprar')
    figura('10_rfm_conversion')
    fig, ax = plt.subplots(figsize=(6, 4))
    for nombre, score in [('R', -rfm.R), ('F', rfm.F), ('M', rfm.M), ('Suma', rfm.suma)]:
        auc = roc_auc_score(rfm.compra, score)
        aucs.append({'dimension': nombre, 'AUC': float(auc)})
        fpr, tpr, _ = roc_curve(rfm.compra, score)
        ax.plot(fpr, tpr, label=f'{nombre}: AUC {auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='#888888')
    ax.set(xlabel='Falsos positivos', ylabel='Verdaderos positivos', title='Capacidad de separar compradores futuros')
    ax.legend()
    figura('11_rfm_auc')
    guardar(pd.DataFrame(conversion), '06_conversion_puntajes')
    guardar(pd.DataFrame(aucs), '06_auc')
    campeones = rfm.index[rfm.campeon]
    nuevos = ~despues.CustomerID.isin(ventas.CustomerID.unique())
    total = despues.monto.sum()
    ingreso_campeones = despues.loc[despues.CustomerID.isin(campeones), 'monto'].sum()
    ingreso_nuevos = despues.loc[nuevos, 'monto'].sum()
    resumen['rfm'] = {'clientes': len(rfm), 'M_no_positivo': int(rfm.M.le(0).sum()),
                       'sin_compra_previa': int(rfm.sin_compra_previa.sum()),
                       'campeones': len(campeones), 'fraccion_campeones': len(campeones)/len(rfm),
                       'venta_futura_conocidos': float(total),
                       'venta_campeones': float(ingreso_campeones),
                       'fraccion_venta_campeones': float(ingreso_campeones/total),
                       'venta_nuevos': float(ingreso_nuevos),
                       'fraccion_venta_nuevos': float(ingreso_nuevos/total),
                       'clientes_nuevos_futuros': int(despues.loc[nuevos, 'CustomerID'].nunique()),
                       'clientes_F1': int(rfm.F.eq(1).sum())}
    guardar(pd.DataFrame([resumen['rfm']]), '06_resumen')
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(['Campeones', 'Otros existentes', 'Nuevos'],
           [ingreso_campeones, total-ingreso_campeones-ingreso_nuevos, ingreso_nuevos])
    ax.set(ylabel='Venta neta (GBP)', title='Origen de las ventas futuras con CustomerID')
    figura('12_origen_venta_futura')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xlsx', type=Path, default=ROOT/'datos'/'Online Retail.xlsx')
    args = parser.parse_args()
    inicio = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    resumen, registro = {}, []
    print('Parte 0: 5000 réplicas por escenario', flush=True)
    parte0(resumen)
    print('Lectura del Excel original y limpieza', flush=True)
    raw = pd.read_excel(args.xlsx, dtype={'InvoiceNo': 'string', 'StockCode': 'string',
                                         'CustomerID': 'string'})
    limpio, serie = partes1y2(raw, resumen)
    print('Partes 3 y 4: estacionalidad y SARIMA', flush=True)
    ma_train = partes3y4(serie, resumen, registro)
    print('Parte 5: validaciones y pronóstico semanal', flush=True)
    parte5(serie, ma_train, resumen, registro)
    print('Parte 6: clientes', flush=True)
    parte6(limpio, resumen)
    (OUT/'convergencia.json').write_text(json.dumps(registro, indent=2, ensure_ascii=False))
    resumen['ejecucion'] = {'segundos': time.perf_counter()-inicio,
                            'sha256_xlsx': hashlib.sha256(args.xlsx.read_bytes()).hexdigest(),
                            'versiones': {p: importlib.metadata.version(p) for p in
                                          ['numpy','pandas','scipy','statsmodels',
                                           'matplotlib','scikit-learn','openpyxl']}}
    (OUT/'resumen.json').write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
