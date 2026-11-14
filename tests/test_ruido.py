"""Testes sem banco de dados: o acesso ao MySQL é substituído por dados sintéticos.

Execute com:  python -m unittest discover -s tests -v
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # permite rodar os testes mesmo sem o driver MySQL instalado
    import mysql.connector  # noqa: F401
except ImportError:
    mc = types.ModuleType("mysql.connector")
    mc.Error = type("Error", (Exception,), {})
    mc.connect = lambda **kw: None
    mysql = types.ModuleType("mysql")
    mysql.connector = mc
    sys.modules["mysql"] = mysql
    sys.modules["mysql.connector"] = mc

import numpy as np  # noqa: E402

import app as app_module  # noqa: E402
import ml_model  # noqa: E402
from geo import Projecao, agrupar_pontos, idw  # noqa: E402

LAT0, LON0 = -23.5505, -46.6333


def campo_sintetico(n=40, seed=0):
    """Fonte de ruído em (LAT0, LON0): nível cai com a distância + pequeno ruído."""
    rng = np.random.default_rng(seed)
    lat = LAT0 + rng.uniform(-0.002, 0.002, n)
    lon = LON0 + rng.uniform(-0.002, 0.002, n)
    xy = Projecao(LAT0, LON0).para_metros(lat, lon)
    d = np.hypot(xy[:, 0], xy[:, 1])
    return lat, lon, 90 - 0.08 * d + rng.normal(0, 0.3, n)


class TestGeo(unittest.TestCase):
    def test_idw_exato_no_sensor_e_dentro_do_intervalo(self):
        xy = np.array([[0, 0], [100, 0], [0, 100]], float)
        v = np.array([60, 70, 80], float)
        self.assertAlmostEqual(idw(xy, v, xy[1:2])[0], 70.0)
        r = idw(xy, v, np.array([[30, 30]], float))[0]
        self.assertTrue(60 <= r <= 80)

    def test_agrupar_duplicados(self):
        lat, lon, v, total = agrupar_pontos([1, 1, 2], [3, 3, 4], [60, 80, 50])
        self.assertEqual((len(v), total), (2, 3))
        self.assertIn(70.0, list(v))


class TestML(unittest.TestCase):
    def test_gpr_supera_media_global_e_incerteza_cresce_longe(self):
        lat, lon, v = campo_sintetico()
        modelo = ml_model.ModeloRuido(lat, lon, v)
        erro_modelo = np.sqrt(np.mean((modelo.validacao_cruzada() - v) ** 2))
        erro_media = np.sqrt(np.mean((v - v.mean()) ** 2))
        self.assertLess(erro_modelo, erro_media)

        _, perto = modelo.prever([LAT0], [LON0])
        _, longe = modelo.prever([LAT0 + 0.05], [LON0 + 0.05])
        self.assertGreater(longe[0], perto[0])


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = app_module.app.test_client()
        self._orig = app_module.carregar_pontos
        self.dados = campo_sintetico()
        app_module.carregar_pontos = lambda metrica: self.dados

    def tearDown(self):
        app_module.carregar_pontos = self._orig

    def test_estimativa_ambos(self):
        r = self.client.get(f"/api/ruido/estimativa?latitude={LAT0}&longitude={LON0}")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertIn("idw", j)
        self.assertIn("ml", j)
        self.assertGreater(j["ml"]["valor_db"], 80)  # perto da fonte
        self.assertIn("coberto_por_sensor", j)

    def test_ponto_longe_nao_coberto(self):
        j = self.client.get(
            f"/api/ruido/estimativa?latitude={LAT0 + 0.05}&longitude={LON0}").get_json()
        self.assertFalse(j["coberto_por_sensor"])

    def test_grade(self):
        r = self.client.get("/api/ruido/grade?linhas=5&colunas=6&metodo=ambos")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertEqual((len(j["ml_db"]), len(j["ml_db"][0])), (5, 6))
        self.assertEqual(len(j["idw_db"]), 5)
        self.assertEqual(len(j["ml_desvio_padrao_db"][0]), 6)

    def test_avaliacao(self):
        j = self.client.get("/api/ruido/avaliacao").get_json()
        self.assertIn("rmse_db", j["idw"])
        self.assertIn("rmse_db", j["ml"])

    def test_poucos_pontos_ml_422_e_idw_ok(self):
        self.dados = tuple(a[:3] for a in self.dados)
        app_module.carregar_pontos = lambda m: self.dados
        q = f"latitude={LAT0}&longitude={LON0}"
        self.assertEqual(self.client.get(f"/api/ruido/estimativa?{q}&metodo=ml").status_code, 422)
        self.assertEqual(self.client.get(f"/api/ruido/estimativa?{q}&metodo=idw").status_code, 200)
        j = self.client.get(f"/api/ruido/estimativa?{q}").get_json()  # ambos
        self.assertIn("erro", j["ml"])

    def test_sem_dados_422(self):
        app_module.carregar_pontos = lambda m: (np.array([]),) * 3
        r = self.client.get(f"/api/ruido/estimativa?latitude={LAT0}&longitude={LON0}")
        self.assertEqual(r.status_code, 422)

    def test_validacoes(self):
        c = self.client
        self.assertEqual(c.get("/api/ruido/estimativa?latitude=95&longitude=0").status_code, 400)
        self.assertEqual(c.get("/api/ruido/estimativa?latitude=abc&longitude=0").status_code, 400)
        self.assertEqual(c.get("/api/ruido/estimativa?longitude=0").status_code, 400)
        self.assertEqual(c.get("/api/ruido/grade?linhas=500").status_code, 400)
        self.assertEqual(c.get("/api/ruido/estimativa?latitude=0&longitude=0&metrica=x").status_code, 400)

    def test_post_exige_lat_lon(self):
        corpo = {"ponderacao_frequencia": "A", "tempo_resposta": "F", "calibracao": "L",
                 "avg_db": 80, "min_db": 70, "max_db": 90, "peak_db": 95}
        r = self.client.post("/api/medicoes", json=corpo)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["campos_ausentes"], ["latitude", "longitude"])
        r = self.client.post("/api/medicoes", json={**corpo, "latitude": 100, "longitude": 0})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
