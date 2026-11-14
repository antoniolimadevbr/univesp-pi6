"""Utilitários geoespaciais: projeção local, agrupamento de pontos, IDW e cobertura.

Tudo aqui é puro (sem banco de dados, sem Flask), o que facilita testar.
"""
import numpy as np

EARTH_RADIUS_M = 6371008.8


class Projecao:
    """Projeção equirretangular local (graus -> metros).

    Adequada para áreas pequenas (bairro, campus, cidade). Para regiões muito
    grandes (centenas de km) prefira uma projeção cartográfica (ex.: UTM).
    """

    def __init__(self, lat0, lon0):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self._cos = np.cos(np.radians(self.lat0))

    def para_metros(self, lat, lon):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        x = np.radians(lon - self.lon0) * EARTH_RADIUS_M * self._cos
        y = np.radians(lat - self.lat0) * EARTH_RADIUS_M
        return np.column_stack([np.ravel(x), np.ravel(y)])


def agrupar_pontos(lat, lon, valores, casas=6):
    """Une medições no mesmo local (média dos valores).

    Retorna (lat, lon, valores, quantidade_original).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    valores = np.asarray(valores, dtype=float)

    grupos = {}
    for la, lo, v in zip(lat, lon, valores):
        grupos.setdefault((round(la, casas), round(lo, casas)), []).append(v)

    chaves = list(grupos.keys())
    return (
        np.array([k[0] for k in chaves]),
        np.array([k[1] for k in chaves]),
        np.array([np.mean(grupos[k]) for k in chaves]),
        len(lat),
    )


def _distancias(a, b):
    """Matriz de distâncias euclidianas (len(a) x len(b))."""
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def distancia_ao_sensor_mais_proximo(xy_sensores, xy_consulta):
    return _distancias(xy_consulta, xy_sensores).min(axis=1)


def idw(xy_sensores, valores, xy_consulta, potencia=2.0):
    """Interpolação por inverso da distância ponderada (IDW / método de Shepard).

    Em um ponto que coincide com um sensor, devolve exatamente o valor medido.
    """
    valores = np.asarray(valores, dtype=float)
    d = _distancias(xy_consulta, xy_sensores)
    exato = d < 1e-9

    with np.errstate(divide="ignore"):
        pesos = 1.0 / np.power(d, potencia)
    pesos[exato.any(axis=1)] = 0.0
    soma = pesos.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        resultado = np.where(soma > 0, (pesos @ valores) / soma, np.nan)

    linhas, colunas = np.where(exato)
    resultado[linhas] = valores[colunas]
    return resultado


def idw_validacao_cruzada(xy, valores, potencia=2.0):
    """Leave-one-out: prediz cada sensor usando apenas os demais."""
    n = len(valores)
    if n < 2:
        raise ValueError("São necessários ao menos 2 pontos para validação cruzada.")
    previsoes = np.empty(n)
    for i in range(n):
        mascara = np.arange(n) != i
        previsoes[i] = idw(xy[mascara], valores[mascara], xy[i:i + 1], potencia)[0]
    return previsoes


def metricas_erro(real, previsto):
    erro = np.asarray(previsto) - np.asarray(real)
    return {
        "rmse_db": float(np.sqrt(np.mean(erro ** 2))),
        "mae_db": float(np.mean(np.abs(erro))),
        "vies_db": float(np.mean(erro)),
    }
