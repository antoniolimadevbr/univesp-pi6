"""Modelo de aprendizado de máquina para estimar ruído onde não há sensores.

Usa Regressão por Processo Gaussiano (GPR) sobre as coordenadas projetadas em
metros. É o mesmo princípio do kriging, mas com hiperparâmetros (escala de
correlação espacial, variância e ruído de medição) aprendidos dos próprios dados.
Além da estimativa, devolve o desvio-padrão previsto: ele cresce longe dos
sensores e indica quão confiável é cada estimativa.
"""
import hashlib
import threading
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from geo import Projecao, metricas_erro


class ModeloRuido:
    def __init__(self, lat, lon, valores):
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        self.valores = np.asarray(valores, dtype=float)

        self.projecao = Projecao(lat.mean(), lon.mean())
        xy = self.projecao.para_metros(lat, lon)
        self.centro = xy.mean(axis=0)
        # Escala isotrópica (mesma para x e y) para não distorcer distâncias.
        self.escala = max(float(xy.std(axis=0).max()), 1.0)
        self.X = (xy - self.centro) / self.escala

        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(length_scale=1.0, length_scale_bounds=(0.05, 20.0))
            + WhiteKernel(noise_level=0.05, noise_level_bounds=(1e-5, 1.0))
        )
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.gpr.fit(self.X, self.valores)

    def _normalizar(self, lat, lon):
        xy = self.projecao.para_metros(lat, lon)
        return (xy - self.centro) / self.escala

    def prever(self, lat, lon):
        """Retorna (média_dB, desvio_padrão_dB) para cada coordenada."""
        media, desvio = self.gpr.predict(self._normalizar(lat, lon), return_std=True)
        return media, desvio

    def parametros(self):
        """Hiperparâmetros aprendidos, com escala espacial em metros."""
        k = self.gpr.kernel_
        comprimento = float(k.k1.k2.length_scale) * self.escala
        return {
            "modelo": "gaussian_process",
            "kernel": "constante * RBF + ruido_branco",
            "escala_correlacao_m": round(comprimento, 1),
            "pontos_treino": int(len(self.valores)),
        }

    def validacao_cruzada(self):
        """Leave-one-out mantendo os hiperparâmetros já aprendidos.

        É uma aproximação levemente otimista (os hiperparâmetros foram
        ajustados com todos os pontos), porém suficiente para comparar com IDW.
        """
        n = len(self.valores)
        previsoes = np.empty(n)
        for i in range(n):
            mascara = np.arange(n) != i
            gpr = GaussianProcessRegressor(
                kernel=self.gpr.kernel_, optimizer=None, normalize_y=True
            )
            gpr.fit(self.X[mascara], self.valores[mascara])
            previsoes[i] = gpr.predict(self.X[i:i + 1])[0]
        return previsoes


_cache = {}
_lock = threading.Lock()


def obter_modelo(metrica, lat, lon, valores):
    """Treina o modelo ou reaproveita o já treinado se os dados não mudaram."""
    dados = np.column_stack([lat, lon, valores]).astype(float)
    assinatura = hashlib.sha1(dados.tobytes()).hexdigest()

    with _lock:
        em_cache = _cache.get(metrica)
        if em_cache and em_cache[0] == assinatura:
            return em_cache[1]
        modelo = ModeloRuido(lat, lon, valores)
        _cache[metrica] = (assinatura, modelo)
        return modelo


def avaliar(modelo):
    return metricas_erro(modelo.valores, modelo.validacao_cruzada())
