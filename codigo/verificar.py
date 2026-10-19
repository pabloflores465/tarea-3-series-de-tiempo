"""Pruebas sobre contabilidad, fechas, cálculos y los resultados de una corrida."""
from pathlib import Path
import json
import sys
import unittest
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from limpieza import limpiar
from estadistica import acf_manual, ljung_box
from modelos import fecha_rezago, errores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'resultados'


class Verificacion(unittest.TestCase):
    def test_acf_y_ljung_box_con_referencia(self):
        y = np.random.default_rng(42).normal(size=300)
        np.testing.assert_allclose(acf_manual(y, 21), acf(y, nlags=21, fft=False)[1:])
        for k in [0, 1, 2]:
            propio = ljung_box(y, 12, k)
            otro = acorr_ljungbox(y, lags=[12], model_df=k).iloc[0]
            np.testing.assert_allclose([propio['Q'], propio['p']], otro, rtol=1e-10)

    def test_limpieza_conserva_devoluciones_anonimos_y_productos_raros(self):
        filas = [
            ['1','12345', 'Producto',2,10.,'10'],
            ['C2','12345','Producto',-1,10.,'10'],
            ['3','DCGSSBOY','BOYS PARTY BAG',1,3.,None],
            ['4','PADS','PADS TO MATCH ALL CUSHIONS',1,.1,'10'],
            ['5','POST','POSTAGE',1,5.,'10'],
            ['6','12345','damaged',-1,0.,None],
            ['A7','B','Adjust bad debt',1,-10.,None],
            ['556444','22502','PICNIC BASKET WICKER 60 PIECES',60,649.5,'15098'],
            ['C556445','M','Manual',-1,38970.,'15098']]
        x = pd.DataFrame(filas, columns=['InvoiceNo','StockCode','Description',
                                        'Quantity','UnitPrice','CustomerID'])
        x['InvoiceDate'] = pd.Timestamp('2011-06-10')
        x['Country'] = 'United Kingdom'
        x = pd.concat([x, x.iloc[[0]]], ignore_index=True)
        original = x.copy(deep=True)
        y = limpiar(x)
        pd.testing.assert_frame_equal(x, original)
        self.assertEqual(set(y.InvoiceNo), {'1','C2','3','4'})
        self.assertEqual(y.CustomerID.isna().sum(), 1)
        self.assertAlmostEqual(y.monto.sum(), 13.1)
        self.assertEqual(len(x)-len(y), y.attrs['bitacora'].filas.sum())
        self.assertAlmostEqual((x.Quantity*x.UnitPrice).sum()-y.monto.sum(),
                               y.attrs['bitacora'].monto_gbp.sum())

    def test_feriados_retroceden_sin_inventar_datos(self):
        s = pd.Series([10., np.nan, np.nan],
                      index=pd.to_datetime(['2011-04-18','2011-04-25','2011-05-02']))
        self.assertEqual(fecha_rezago(s, '2011-05-09', 1), pd.Timestamp('2011-04-18'))
        self.assertEqual(fecha_rezago(s, '2011-05-09', 2), pd.Timestamp('2011-04-18'))

    def test_fechas_y_fuentes_no_tienen_fuga(self):
        x = pd.read_csv(OUT/'05_fuentes_rezagos.csv', parse_dates=['origen','objetivo','fuente'])
        self.assertTrue((x.fuente <= x.origen).all())
        self.assertTrue((x.origen < x.objetivo).all())
        self.assertTrue(((x.objetivo-x.origen).dt.days <= 7).all())
        self.assertTrue(x.origen.dt.weekday.eq(6).all())
        p = pd.read_csv(OUT/'05_predicciones.csv', parse_dates=['origen','fecha'])
        self.assertTrue(np.isfinite(p[['real','pred']]).all().all())
        self.assertFalse(p.fecha.eq(pd.Timestamp('2011-10-02')).any())
        self.assertLessEqual(p.loc[p.periodo.eq('validacion'),'fecha'].max(), pd.Timestamp('2011-09-30'))
        self.assertEqual(p.loc[p.periodo.eq('prueba'),'fecha'].max(), pd.Timestamp('2011-12-08'))
        self.assertFalse(((p.modelo=='TRAMPA') & (p.periodo=='prueba')).any())
        for periodo, g in p.loc[p.modelo.ne('TRAMPA')].groupby('periodo'):
            base = set(g.loc[g.modelo.eq('B1'), 'fecha'])
            for _, gg in g.groupby('modelo'):
                self.assertEqual(set(gg.fecha), base)
                self.assertEqual(len(gg), len(base))

    def test_metricas_se_calculan_sobre_todos_los_dias(self):
        p = pd.read_csv(OUT/'05_predicciones.csv')
        tabla = pd.read_csv(OUT/'05_comparacion_modelos.csv').set_index('modelo')
        for (periodo, modelo), g in p.groupby(['periodo','modelo']):
            clave = 'b' if periodo=='validacion' else 'c'
            for metrica, valor in errores(g.real, g.pred).items():
                self.assertAlmostEqual(valor, tabla.loc[modelo, clave+'_'+metrica], places=4)

    def test_convergencia_en_cada_origen(self):
        x = json.loads((OUT/'convergencia.json').read_text())
        finales = {(r['origen'],r['modelo']): r for r in x}
        self.assertEqual(len(finales), 66)  # 22 orígenes + 10 de prueba + ajuste inicial, por dos
        self.assertTrue(all(r['convergio'] for r in finales.values()))
        self.assertTrue(all(r['maxiter']==500 for key,r in finales.items() if key[0]!='ajuste_inicial'))

    def test_rfm_etiquetas_y_denominadores(self):
        x = pd.read_csv(OUT/'06_clientes_rfm.csv')
        res = json.loads((OUT/'resumen.json').read_text())['rfm']
        self.assertEqual(len(x), res['clientes'])
        self.assertEqual(int(x.M.le(0).sum()), res['M_no_positivo'])
        self.assertTrue(x[['puntaje_R','puntaje_F','puntaje_M']].isin(range(1,6)).all().all())
        np.testing.assert_array_equal(x.campeon, x[['puntaje_R','puntaje_F','puntaje_M']].ge(4).all(axis=1))
        self.assertAlmostEqual(res['venta_campeones']/res['venta_futura_conocidos'],
                               res['fraccion_venta_campeones'])
        self.assertAlmostEqual(res['venta_nuevos']/res['venta_futura_conocidos'],
                               res['fraccion_venta_nuevos'])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Verificacion)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidencia = {'pruebas': result.testsRun, 'fallos': len(result.failures),
                 'errores': len(result.errors), 'correcto': result.wasSuccessful(),
                 'comando': 'python codigo/verificar.py',
                 'comprueba': ['ACF y Ljung-Box contra statsmodels',
                              'Conservación de devoluciones y ventas anónimas',
                              'Balance de filas y montos de limpieza',
                              'Rezagos alrededor de feriados',
                              'Fuentes de rezagos anteriores al origen',
                              'Mismas fechas evaluadas en todos los modelos legales',
                              'Métricas sobre todos los días pronosticados',
                              'Convergencia final de los 66 ajustes SARIMA',
                              'Puntajes RFM y denominadores monetarios']}
    (OUT/'verificacion.json').write_text(json.dumps(evidencia, indent=2, ensure_ascii=False)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
