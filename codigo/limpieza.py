"""Una función pública: limpiar(df). No elimina clientes anónimos ni devoluciones."""
import pandas as pd


def limpiar(df):
    """Devuelve una copia limpia; la bitácora queda en resultado.attrs['bitacora'].

    Las reglas se aplican en el orden mostrado. Monto = Quantity * UnitPrice,
    con signo: una devolución debe restar. No se recortan montos extremos.
    PADS y DCGS son productos reales según sus descripciones. Los vouchers,
    portes, ajustes manuales y cargos administrativos no son mercadería.
    """
    x = df.copy()
    originales = list(x.columns)
    for col in ['InvoiceNo', 'StockCode', 'CustomerID']:
        x[col] = x[col].astype('string')
    x['InvoiceDate'] = pd.to_datetime(x.InvoiceDate)
    bitacora = []

    def quitar(mascara, regla):
        nonlocal x
        bitacora.append({'regla': regla, 'filas': int(mascara.sum()),
                         'monto_gbp': float((x.loc[mascara, 'Quantity'] *
                                              x.loc[mascara, 'UnitPrice']).sum())})
        x = x.loc[~mascara].copy()

    quitar(x.duplicated(subset=originales), '1. Duplicados exactos')
    # C = cancelación válida; A = ajuste contable, no factura comercial.
    quitar(~x.InvoiceNo.str.match(r'^(\d|C)', na=False), '2. Prefijo no comercial')
    # Excepción documentada: pedido erróneo de 60 cestas, anulado tres minutos
    # después mediante un crédito M por exactamente el mismo monto.
    # Se eliminan AMBOS lados antes de excluir los ajustes M en general.
    quitar(x.InvoiceNo.isin(['556444', 'C556445']), '3. Pedido anulado y crédito manual pareado')
    producto = (x.StockCode.str.match(r'^\d{5}', na=False) |
                x.StockCode.str.startswith('DCGS', na=False) | x.StockCode.eq('PADS'))
    quitar(~producto, '4. Cargos, ajustes y vouchers')
    quitar(x.UnitPrice.le(0) | x.UnitPrice.isna(), '5. Precio no positivo')
    quitar(x.Quantity.lt(0) & ~x.InvoiceNo.str.startswith('C'),
           '6. Salida de inventario sin cancelación')
    quitar(x.Quantity.eq(0) | x.Quantity.isna(), '7. Cantidad nula')
    # No cambiar el signo de una cancelación: ya viene negativo en este archivo.
    if (x.InvoiceNo.str.startswith('C') & x.Quantity.ge(0)).any():
        raise ValueError('Cancelación con cantidad no negativa: revisar el origen.')
    x['monto'] = x.Quantity * x.UnitPrice
    x['fecha'] = x.InvoiceDate.dt.normalize()
    x.attrs['bitacora'] = pd.DataFrame(bitacora)
    return x
