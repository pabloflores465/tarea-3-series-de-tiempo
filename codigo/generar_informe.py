"""Actualiza las tablas y cifras LaTeX desde los resultados; no copia cifras a mano."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'resultados'
DOC = ROOT/'informe'
TAB = DOC/'tablas'


def esc(s):
    return str(s).replace('&', r'\&').replace('_', r'\_').replace('%', r'\%').replace('#', r'\#')


def n(x, dec=2):
    return f'{float(x):,.{dec}f}'.replace(',', r'\,')


def pct(x):
    return n(100*float(x), 2) + r'\%'


def pm(x, se, dec=4):
    return '$' + n(x,dec) + r'\pm ' + n(se,dec) + '$'


def tabla(nombre, encabezados, filas, especificacion=None):
    especificacion = especificacion or 'l' + 'r'*(len(encabezados)-1)
    text = r'\begin{center}\small\setlength{\tabcolsep}{4pt}'+'\n'
    text += r'\begin{adjustbox}{max width=\textwidth}\begin{tabular}{'+especificacion+'}\n'+r'\toprule'+'\n'
    text += ' & '.join(encabezados) + r' \\ \midrule'+'\n'
    text += '\n'.join(' & '.join(map(str,row)) + r' \\' for row in filas)
    text += '\n'+r'\bottomrule\end{tabular}\end{adjustbox}\end{center}'+'\n'
    (TAB/(nombre+'.tex')).write_text(text)


def main():
    TAB.mkdir(exist_ok=True)
    # El proyecto LaTeX también funciona sin acceso a carpetas superiores.
    shutil.copytree(OUT/'figuras', DOC/'figuras', dirs_exist_ok=True)
    s = json.loads((OUT/'resumen.json').read_text())
    load = lambda name: pd.read_csv(OUT/(name+'.csv'))
    macros = {}
    def macro(k,v): macros[k] = str(v)
    for key, field in [('FilasOriginal','filas'),('FacturasOriginal','facturas'),
                       ('ProductosOriginal','productos'),('ClientesOriginal','clientes'),
                       ('SinClienteOriginal','sin_cliente')]:
        macro(key,n(s['original'][field],0))
    macro('PorcentajeAnonimos',pct(s['original']['fraccion_sin_cliente']))
    macro('FilasLimpias',n(s['limpio']['filas'],0))
    macro('MontoLimpio',n(s['limpio']['monto']))
    macro('Devoluciones',n(s['limpio']['devoluciones'],0))
    macro('PromedioTrain',n(s['medias']['entrenamiento']))
    macro('PromedioTest',n(s['medias']['prueba']))
    macro('PhiResiduos',n(s['ar_residuos']['phi'],4))
    macro('ConstanteResiduos',n(s['ar_residuos']['c'],2))
    macro('GanadorMAE',s['ganadores']['MAE'])
    macro('GanadorRMSE',s['ganadores']['RMSE'])
    macro('LagsOriginal',', '.join(map(str,s['acf_original']['fuera'])))
    macro('NumeroLags',len(s['acf_original']['fuera']))
    acf = load('03_04_acf')
    for nombre, campo, tam in [('LagsIndicadoras','Residuo de indicadoras',245),
                              ('LagsAR','Residuo de AR(1)',244)]:
        macro(nombre, ', '.join(map(str, acf.loc[acf[campo].abs()>1.96/np.sqrt(tam),'lag'])))
    mc = load('00_simulacion')
    tabla('simulacion', ['Escenario',r'$\bar r_1\pm EE$',r'$\bar r_5\pm EE$',
                         r'$\phi^5$',r'$\bar r_7\pm EE$',r'$\overline{\hat\phi}\pm EE$'],
          [[esc(r.escenario),pm(r.r1,r.r1_se),pm(r.r5,r.r5_se),
            '--' if pd.isna(r.phi5_referencia) else n(r.phi5_referencia,5),
            pm(r.r7,r.r7_se),pm(r.phi_ols,r.phi_ols_se)] for r in mc.itertuples()])
    tabla('rechazos',['Escenario','Semilla',r'Rechazo $r_1$ $\pm EE$',r'Rechazo LB(10) $\pm EE$'],
          [[esc(r.escenario),r.semilla,pm(r.rechazo_r1,r.rechazo_r1_se),
            pm(r.rechazo_lb10,r.rechazo_lb10_se)] for r in mc.itertuples()])
    b = load('01_bitacora')
    tabla('bitacora',['Regla (en orden)','Filas eliminadas','Monto firmado (GBP)'],
          [[esc(r.regla),n(r.filas,0),n(r.monto_gbp,3)] for r in b.itertuples()],
          r'p{8cm}rr')
    top = load('01_seis_extremos')
    tabla('extremos',['Factura','Fecha','Producto','Cantidad','Precio','Monto (GBP)','Queda'],
          [[r.InvoiceNo,str(r.InvoiceDate)[:10],r.StockCode,n(r.Quantity,0),n(r.UnitPrice),
            n(r.monto),'Sí' if r.se_conserva else 'No'] for r in top.itertuples()])
    cierres = load('02_dias_sin_transacciones')
    cierres['fecha'] = pd.to_datetime(cierres.fecha)
    sab = cierres.loc[cierres.fecha.dt.weekday.eq(5)].copy()
    sab['mes'] = sab.fecha.dt.strftime('%Y-%m')
    tabla('sabados',['Mes','Sábados sin transacciones (día del mes)'],
          [[mes,', '.join(g.fecha.dt.strftime('%d'))] for mes,g in sab.groupby('mes')], 'll')
    otro = cierres.loc[~cierres.fecha.dt.weekday.eq(5)]
    tabla('otros_cierres',['Fecha','Día','Clasificación'],
          [[r.fecha.strftime('%Y-%m-%d'),r.dia,esc(r.clasificacion)] for r in otro.itertuples()],
          r'llp{8cm}')
    est = load('03_indices_estacionales')
    tabla('estacionales',['Día','Índice','Cocientes válidos'],
          [[r.dia,n(r.indice,4),r.n] for r in est.itertuples()])
    macro('PromedioIndices',n(est.indice.mean(),4))
    diag = load('04_ljung_box')
    tabla('diagnostico',['Serie / residuo','n','m','Q','gl','p'],
          [[r.modelo,r.n,r.m,n(r.Q,3),r.gl,f'${r.p:.3g}$'] for r in diag.itertuples()])
    params = load('04_parametros_sarima')
    tabla('parametros',['Modelo','Parámetro','Estimación'],
          [[r.modelo,esc(r.parametro),n(r.valor,5)] for r in params.itertuples()])
    reg = json.loads((OUT/'convergencia.json').read_text())
    inicial = [r for r in reg if r['origen']=='ajuste_inicial']
    tabla('convergencia_inicial',['Modelo','Límite','Convergió',r'$c$',r'$\phi / \theta$',r'$\Phi / \Theta$',r'$\sigma^2$'],
          [[r['modelo'],r['maxiter'],'Sí' if r['convergio'] else 'No',
            n(r['parametros'].get('intercept',0),5),
            n(r['parametros'].get('ar.L1',r['parametros'].get('ma.L1')),5),
            n(r['parametros'].get('ar.S.L6',r['parametros'].get('ma.S.L6')),5),
            n(r['parametros']['sigma2'],5)] for r in inicial])
    finales = {(r['origen'],r['modelo']): r for r in reg}
    macro('AjustesConvergentes',sum(r['convergio'] for r in finales.values()))
    macro('IntentosNoConvergentes',sum(not r['convergio'] for r in reg))
    comparacion = load('05_comparacion_modelos')
    filas=[]
    for r in comparacion.itertuples():
        row=[r.modelo]
        for k in ['MAE','RMSE']:
            row.append('--' if pd.isna(getattr(r,'a_'+k)) else pm(getattr(r,'a_'+k),getattr(r,'a_'+k+'_sd'),0))
        for clave in ['b_MAE','b_RMSE','c_MAE','c_RMSE']:
            row.append('--' if pd.isna(getattr(r,clave)) else n(getattr(r,clave),0))
        filas.append(row)
    tabla('comparacion',['Modelo',r'(a) MAE $\pm$ DE',r'(a) RMSE $\pm$ DE',
                         '(b) MAE','(b) RMSE','(c) MAE','(c) RMSE'], filas)
    tabla('muestras',['Modelo','n de CV','n de validación','n de prueba'],
          [[r.modelo,r.a_n,r.b_n,r.c_n] for r in comparacion.itertuples()])
    best = comparacion.set_index('modelo').loc[s['ganadores']['MAE']]
    macro('MejorMAE',n(best.c_MAE))
    macro('MejorRMSE',n(comparacion.set_index('modelo').loc[s['ganadores']['RMSE'],'c_RMSE']))
    conv=load('06_conversion_puntajes')
    filas=[]
    for i in range(1,6):
        row=[i]
        for dim in ['R','F','M']:
            r=conv.loc[(conv.dimension==dim)&(conv.puntaje==i)].iloc[0]
            row += [f'{int(r.compraron)}/{int(r.clientes)}',pct(r.fraccion_compra)]
        filas.append(row)
    tabla('conversion',['Puntaje','R: compran/total','R: fracción','F: compran/total','F: fracción',
                        'M: compran/total','M: fracción'],filas)
    auc=load('06_auc')
    tabla('auc',['Clasificador','AUC'],[[r.dimension,n(r.AUC,4)] for r in auc.itertuples()])
    macro('MejorDimension',auc.loc[auc.dimension.ne('Suma')].sort_values('AUC').iloc[-1].dimension)
    rr=s['rfm']
    for clave,campo in [('ClientesRFM','clientes'),('ClientesNoPositivos','M_no_positivo'),
                         ('SoloDevoluciones','sin_compra_previa'),('Campeones','campeones'),
                         ('Nuevos','clientes_nuevos_futuros'),('FrecuenciaUno','clientes_F1')]:
        macro(clave,n(rr[campo],0))
    for clave,campo in [('PorcentajeCampeones','fraccion_campeones'),
                         ('PorcentajeVentaCampeones','fraccion_venta_campeones'),
                         ('PorcentajeNuevos','fraccion_venta_nuevos')]:
        macro(clave,pct(rr[campo]))
    for clave,campo in [('VentaFutura','venta_futura_conocidos'),('VentaCampeones','venta_campeones'),
                         ('VentaNuevos','venta_nuevos')]:
        macro(clave,n(rr[campo]))
    tabla('segmentos',['Medida','Resultado'],[
        ['Clientes históricos',n(rr['clientes'],0)],
        ['Campeones',n(rr['campeones'],0)+' ('+pct(rr['fraccion_campeones'])+')'],
        ['Venta futura de todos con ID (GBP)',n(rr['venta_futura_conocidos'])],
        ['Venta futura de campeones (GBP)',n(rr['venta_campeones'])],
        ['Fracción de venta de campeones',pct(rr['fraccion_venta_campeones'])],
        ['Clientes sin venta antes del corte',n(rr['clientes_nuevos_futuros'],0)],
        ['Venta futura de nuevos (GBP)',n(rr['venta_nuevos'])],
        ['Fracción de venta de nuevos',pct(rr['fraccion_venta_nuevos'])]],'lr')
    macro('HashDatos',s['ejecucion']['sha256_xlsx'])
    # Compara la trampa en las MISMAS fechas para aislar el efecto del predictor.
    pred=load('05_predicciones')
    trampa=pred.loc[pred.modelo.eq('TRAMPA')]
    m2=pred.loc[(pred.modelo=='M2')&(pred.periodo=='validacion')]
    par=trampa[['fecha','real','pred']].merge(m2[['fecha','pred']],on='fecha',suffixes=('_trampa','_m2'))
    rows=[]
    for nombre,col in [('M2, mismas fechas','pred_m2'),('TRAMPA','pred_trampa')]:
        err=par.real-par[col]
        rows.append([nombre,len(par),n(err.abs().mean()),n(np.sqrt((err**2).mean()))])
    tabla('trampa_comparable',['Modelo','n','MAE (b)','RMSE (b)'],rows)
    (DOC/'cifras.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
    print(f'Informe actualizado: {len(macros)} cifras y {len(list(TAB.glob("*.tex")))} tablas.')


if __name__=='__main__':
    main()
