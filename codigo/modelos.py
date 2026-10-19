"""Modelos sencillos y reajuste semanal: los datos futuros no entran al ajuste."""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from statsmodels.tsa.statespace.sarimax import SARIMAX

DIAS = [0, 1, 2, 3, 4, 6]  # lunes a viernes y domingo
CONFIG = {'S1': dict(order=(1, 0, 0), seasonal_order=(1, 0, 0, 6), trend='c'),
          'S2': dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, 6), trend='n')}


def indicadoras(fechas):
    return np.column_stack([(fechas.weekday == dia).astype(float) for dia in DIAS])


def fecha_rezago(serie, fecha, semanas):
    """Retrocede semanas desde fecha-semanas*7 hasta encontrar un día observado."""
    candidato = pd.Timestamp(fecha) - pd.Timedelta(days=7 * semanas)
    while candidato >= serie.index.min():
        if candidato in serie.index and pd.notna(serie.loc[candidato]):
            return candidato
        candidato -= pd.Timedelta(days=7)
    return pd.NaT


def diseno(serie, fechas, modelo, media_centrada=None):
    x = indicadoras(fechas)
    if modelo == 'M1':
        t = (fechas - serie.index.min()).days.to_numpy() / 365
        x = np.column_stack([x, t])
    elif modelo in ['M2', 'TRAMPA']:
        rezagos = []
        for fecha in fechas:
            fuentes = [fecha_rezago(serie, fecha, s) for s in [1, 2]]
            rezagos.append([serie.loc[f] if pd.notna(f) else np.nan for f in fuentes])
        x = np.column_stack([x, rezagos])
        if modelo == 'TRAMPA':
            x = np.column_stack([x, media_centrada.reindex(fechas).to_numpy()])
    return x


def ajustar_sarima(valores, modelo, registro, origen):
    """Misma configuración y secuencia de reintentos en todos los orígenes."""
    if np.any(np.asarray(valores) <= 0):
        raise ValueError('No se puede tomar log de venta no positiva.')
    m = SARIMAX(np.log(np.asarray(valores)), **CONFIG[modelo],
                enforce_stationarity=True, enforce_invertibility=True)
    anterior = None
    pasos = [('lbfgs', 500), ('powell', 500)]
    if origen == 'ajuste_inicial':
        pasos = [('lbfgs', 50)] + pasos
    for metodo, limite in pasos:
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter('always')
            r = m.fit(disp=False, method=metodo, maxiter=limite,
                      start_params=anterior)
        convergio = bool(r.mle_retvals.get('converged', False))
        registro.append({'origen': str(origen), 'modelo': modelo,
                         'metodo': metodo, 'maxiter': limite, 'convergio': convergio,
                         'loglike': float(r.llf),
                         'parametros': dict(zip(r.param_names, r.params.tolist())),
                         'avisos': [str(a.message) for a in avisos]})
        if convergio and np.isfinite(r.params).all():
            return r
        anterior = r.params
    raise RuntimeError(f'{modelo} no convergió en {origen}. No se omite el origen.')


def errores(real, pred):
    e = np.asarray(real) - np.asarray(pred)
    return {'MAE': float(np.abs(e).mean()), 'RMSE': float(np.sqrt((e * e).mean()))}


def cv_aleatoria(serie, media_centrada):
    """Diez semillas, cinco folds; concatena predicciones antes de medir error."""
    train = serie.loc[:'2011-09-30'].dropna()
    filas = []
    for modelo in ['B1', 'M1', 'M2', 'TRAMPA']:
        x = diseno(serie, train.index, modelo, media_centrada)
        valido = np.isfinite(x).all(axis=1)
        xx, yy = x[valido], train.to_numpy()[valido]
        for semilla in range(2026, 2036):
            pred = np.zeros(len(yy))
            for a, b in KFold(5, shuffle=True, random_state=semilla).split(xx):
                if modelo == 'B1':
                    pred[b] = yy[a].mean()
                else:
                    beta = np.linalg.lstsq(xx[a], yy[a], rcond=None)[0]
                    pred[b] = xx[b] @ beta
            filas.append({'modelo': modelo, 'semilla': semilla, 'n': len(yy),
                          **errores(yy, pred)})
    return pd.DataFrame(filas)


def origen_movil(serie, media_centrada, periodo, registro):
    if periodo == 'validacion':
        origenes = pd.date_range('2011-05-01', '2011-09-25', freq='W-SUN')
        final = pd.Timestamp('2011-09-30')
    else:
        origenes = pd.date_range('2011-10-02', '2011-12-04', freq='W-SUN')
        final = pd.Timestamp('2011-12-08')
    resultados, fuentes = [], []
    for origen in origenes:
        historia = serie.loc[:origen].dropna()
        objetivos = serie.loc[origen + pd.Timedelta(days=1):
                              min(origen + pd.Timedelta(days=7), final)].dropna()
        assert historia.index.max() <= origen < objetivos.index.min()
        # Todo el vector de la semana se calcula antes de ver sus ventas.
        pronosticos = {'B1': np.repeat(historia.mean(), len(objetivos)),
                      'B2': np.array([historia.loc[historia.index.weekday == d.weekday()].iloc[-1]
                                      for d in objetivos.index])}
        for modelo in ['M1', 'M2'] + (['TRAMPA'] if periodo == 'validacion' else []):
            x = diseno(serie, historia.index, modelo, media_centrada)
            xx = diseno(serie, objetivos.index, modelo, media_centrada)
            valido = np.isfinite(x).all(axis=1)
            beta = np.linalg.lstsq(x[valido], historia.to_numpy()[valido], rcond=None)[0]
            pronosticos[modelo] = xx @ beta
        for nombre in ['S1', 'S2']:
            fit = ajustar_sarima(historia.to_numpy(), nombre, registro, origen.date())
            pronosticos[nombre] = np.exp(fit.forecast(len(objetivos)))
        for fecha in objetivos.index:
            for semanas in [1, 2]:
                fuente = fecha_rezago(serie, fecha, semanas)
                assert pd.notna(fuente) and fuente <= origen
                fuentes.append({'periodo': periodo, 'origen': origen, 'objetivo': fecha,
                                'semanas': semanas, 'fuente': fuente})
        for nombre, pred in pronosticos.items():
            for fecha, real, p in zip(objetivos.index, objetivos, pred):
                if not np.isfinite(p):
                    # Media centrada: sus tres últimas fechas no tienen ventana
                    # completa dentro del entrenamiento. No rellenar con prueba.
                    assert nombre == 'TRAMPA'
                    continue
                resultados.append({'periodo': periodo, 'origen': origen, 'fecha': fecha,
                                   'modelo': nombre, 'real': float(real), 'pred': float(p),
                                   'n_historia': len(historia)})
        print(f'  {periodo}: origen {origen.date()}, {len(objetivos)} objetivos', flush=True)
    return pd.DataFrame(resultados), pd.DataFrame(fuentes)
