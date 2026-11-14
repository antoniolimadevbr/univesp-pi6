import math
import os
from decimal import Decimal, InvalidOperation

import mysql.connector
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from mysql.connector import Error

import ml_model
from geo import (
    Projecao,
    agrupar_pontos,
    distancia_ao_sensor_mais_proximo,
    idw,
    idw_validacao_cruzada,
    metricas_erro,
)

load_dotenv()

app = Flask(__name__)

REQUIRED_FIELDS = {
    "ponderacao_frequencia",
    "tempo_resposta",
    "calibracao",
    "avg_db",
    "min_db",
    "max_db",
    "peak_db",
    "latitude",
    "longitude",
}

METRICAS_DB = ("avg_db", "min_db", "max_db", "peak_db")
COORD_QUANTUM = Decimal("0.000001")
MAX_CELULAS_GRADE = 10_000

RAIO_COBERTURA_M = float(os.getenv("RAIO_COBERTURA_M", "50"))
MIN_PONTOS_ML = int(os.getenv("MIN_PONTOS_ML", "5"))


class ErroDeRequisicao(Exception):
    def __init__(self, mensagem, status=400):
        super().__init__(mensagem)
        self.mensagem = mensagem
        self.status = status


@app.errorhandler(ErroDeRequisicao)
def tratar_erro_requisicao(exc):
    return jsonify({"erro": exc.mensagem}), exc.status


@app.errorhandler(Error)
def tratar_erro_mysql(exc):
    app.logger.exception("Erro no MySQL")
    return jsonify({"erro": "Não foi possível consultar as medições."}), 500


def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "medicoes_db"),
    )


def parse_decimal(value, field):
    try:
        number = Decimal(str(value).replace(",", "."))
    except (InvalidOperation, ValueError):
        raise ValueError(f"O campo '{field}' deve ser numérico.")

    if not number.is_finite():
        raise ValueError(f"O campo '{field}' deve ser um número finito.")

    return number


def parse_coordenada(value, field, limite):
    number = parse_decimal(value, field)
    if not (-limite <= number <= limite):
        raise ValueError(f"O campo '{field}' deve estar entre -{limite} e {limite}.")
    return number.quantize(COORD_QUANTUM)


def _serializar_medicao(row):
    for field in ("avg_db", "min_db", "max_db", "peak_db"):
        row[field] = float(row[field])
    for field in ("latitude", "longitude"):
        # Medições antigas (anteriores à migração) podem não ter coordenadas.
        row[field] = float(row[field]) if row[field] is not None else None
    if row["criado_em"] is not None:
        row["criado_em"] = row["criado_em"].isoformat()
    return row


COLUNAS_SELECT = """
    id, ponderacao_frequencia, tempo_resposta, calibracao,
    avg_db, min_db, max_db, peak_db, latitude, longitude, criado_em
"""


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/api/medicoes")
def criar_medicao():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"erro": "Envie um objeto JSON válido."}), 400

    missing = sorted(REQUIRED_FIELDS - data.keys())
    if missing:
        return jsonify({
            "erro": "Campos obrigatórios ausentes.",
            "campos_ausentes": missing
        }), 400

    try:
        ponderacao = str(data["ponderacao_frequencia"]).strip()
        tempo_resposta = str(data["tempo_resposta"]).strip()
        calibracao = str(data["calibracao"]).strip()

        if not ponderacao or not tempo_resposta or not calibracao:
            return jsonify({"erro": "Os campos de texto não podem ficar vazios."}), 400

        avg_db = parse_decimal(data["avg_db"], "avg_db")
        min_db = parse_decimal(data["min_db"], "min_db")
        max_db = parse_decimal(data["max_db"], "max_db")
        peak_db = parse_decimal(data["peak_db"], "peak_db")
        latitude = parse_coordenada(data["latitude"], "latitude", 90)
        longitude = parse_coordenada(data["longitude"], "longitude", 180)

        if min_db > max_db:
            return jsonify({"erro": "min_db não pode ser maior que max_db."}), 400

        if not (min_db <= avg_db <= max_db):
            return jsonify({"erro": "avg_db deve estar entre min_db e max_db."}), 400

        if peak_db < max_db:
            return jsonify({"erro": "peak_db não pode ser menor que max_db."}), 400

    except ValueError as exc:
        return jsonify({"erro": str(exc)}), 400

    sql = """
        INSERT INTO medicoes (
            ponderacao_frequencia, tempo_resposta, calibracao,
            avg_db, min_db, max_db, peak_db, latitude, longitude
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(sql, (
            ponderacao, tempo_resposta, calibracao,
            avg_db, min_db, max_db, peak_db, latitude, longitude
        ))
        connection.commit()
        measurement_id = cursor.lastrowid

        return jsonify({
            "mensagem": "Medição armazenada com sucesso.",
            "id": measurement_id
        }), 201

    except Error:
        app.logger.exception("Erro ao salvar medição no MySQL")
        return jsonify({"erro": "Não foi possível armazenar a medição."}), 500

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


@app.get("/api/medicoes")
def listar_medicoes():
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(f"""
            SELECT {COLUNAS_SELECT}
            FROM medicoes
            ORDER BY criado_em DESC
        """)
        rows = [_serializar_medicao(row) for row in cursor.fetchall()]
        return jsonify(rows), 200

    except Error:
        app.logger.exception("Erro ao consultar medições no MySQL")
        return jsonify({"erro": "Não foi possível consultar as medições."}), 500

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


@app.get("/api/medicoes/<int:measurement_id>")
def obter_medicao(measurement_id):
    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(f"""
            SELECT {COLUNAS_SELECT}
            FROM medicoes
            WHERE id = %s
        """, (measurement_id,))
        row = cursor.fetchone()

        if row is None:
            return jsonify({"erro": "Medição não encontrada."}), 404

        return jsonify(_serializar_medicao(row)), 200

    except Error:
        app.logger.exception("Erro ao consultar medição no MySQL")
        return jsonify({"erro": "Não foi possível consultar a medição."}), 500

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()


# ---------------------------------------------------------------------------
# Interpolação espacial e estimativa por aprendizado de máquina
# ---------------------------------------------------------------------------

def carregar_pontos(metrica):
    """Lê do banco (latitude, longitude, valor) das medições com coordenadas."""
    if metrica not in METRICAS_DB:  # whitelist: o nome da coluna vai no SQL
        raise ErroDeRequisicao(f"'metrica' deve ser uma de: {', '.join(METRICAS_DB)}.")

    connection = None
    cursor = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute(f"""
            SELECT latitude, longitude, {metrica}
            FROM medicoes
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """)
        linhas = cursor.fetchall()
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None and connection.is_connected():
            connection.close()

    if not linhas:
        return np.array([]), np.array([]), np.array([])
    arr = np.array([[float(c) for c in linha] for linha in linhas])
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _arg_float(nome, padrao=None, minimo=None, maximo=None):
    bruto = request.args.get(nome)
    if bruto is None or bruto == "":
        if padrao is None:
            raise ErroDeRequisicao(f"O parâmetro '{nome}' é obrigatório.")
        return padrao
    try:
        valor = float(bruto.replace(",", "."))
    except ValueError:
        raise ErroDeRequisicao(f"O parâmetro '{nome}' deve ser numérico.")
    if not math.isfinite(valor):
        raise ErroDeRequisicao(f"O parâmetro '{nome}' deve ser finito.")
    if (minimo is not None and valor < minimo) or (maximo is not None and valor > maximo):
        raise ErroDeRequisicao(f"O parâmetro '{nome}' deve estar entre {minimo} e {maximo}.")
    return valor


def _arg_int(nome, padrao, minimo, maximo):
    valor = _arg_float(nome, float(padrao), minimo, maximo)
    if valor != int(valor):
        raise ErroDeRequisicao(f"O parâmetro '{nome}' deve ser inteiro.")
    return int(valor)


def _contexto():
    """Lê metrica/metodo/potencia e prepara os pontos (agrupados) e a projeção."""
    metrica = request.args.get("metrica", "avg_db")
    if metrica not in METRICAS_DB:
        raise ErroDeRequisicao(f"'metrica' deve ser uma de: {', '.join(METRICAS_DB)}.")
    metodo = request.args.get("metodo", "ambos").lower()
    if metodo not in ("idw", "ml", "ambos"):
        raise ErroDeRequisicao("'metodo' deve ser 'idw', 'ml' ou 'ambos'.")
    potencia = _arg_float("potencia", 2.0, 0.5, 6.0)

    lat, lon, valores = carregar_pontos(metrica)
    if len(valores) == 0:
        raise ErroDeRequisicao(
            "Não há medições com latitude/longitude para interpolar.", 422)

    lat, lon, valores, total = agrupar_pontos(lat, lon, valores)
    projecao = Projecao(lat.mean(), lon.mean())
    xy = projecao.para_metros(lat, lon)

    ml_disponivel = len(valores) >= MIN_PONTOS_ML
    if metodo == "ml" and not ml_disponivel:
        raise ErroDeRequisicao(
            f"O modelo de ML precisa de ao menos {MIN_PONTOS_ML} locais distintos "
            f"com medição (há {len(valores)}). Use metodo=idw.", 422)

    return {
        "metrica": metrica, "metodo": metodo, "potencia": potencia,
        "lat": lat, "lon": lon, "valores": valores, "xy": xy,
        "projecao": projecao, "total_medicoes": total,
        "locais": len(valores), "ml_disponivel": ml_disponivel,
    }


def _estimar(ctx, lat_q, lon_q):
    """Calcula IDW e/ou ML nos pontos de consulta (arrays 1-D)."""
    xy_q = ctx["projecao"].para_metros(lat_q, lon_q)
    dist = distancia_ao_sensor_mais_proximo(ctx["xy"], xy_q)
    saida = {"distancia": dist}

    if ctx["metodo"] in ("idw", "ambos"):
        saida["idw"] = idw(ctx["xy"], ctx["valores"], xy_q, ctx["potencia"])

    if ctx["metodo"] in ("ml", "ambos") and ctx["ml_disponivel"]:
        modelo = ml_model.obter_modelo(
            ctx["metrica"], ctx["lat"], ctx["lon"], ctx["valores"])
        media, desvio = modelo.prever(lat_q, lon_q)
        saida["ml"] = media
        saida["ml_desvio"] = desvio
        saida["ml_parametros"] = modelo.parametros()
    return saida


def _r2(x):
    return round(float(x), 2)


@app.get("/api/ruido/estimativa")
def estimar_ponto():
    """Estima o ruído em uma coordenada qualquer (com ou sem sensor)."""
    latitude = _arg_float("latitude", minimo=-90, maximo=90)
    longitude = _arg_float("longitude", minimo=-180, maximo=180)
    ctx = _contexto()

    try:
        est = _estimar(ctx, np.array([latitude]), np.array([longitude]))
    except Error:
        app.logger.exception("Erro ao consultar medições no MySQL")
        return jsonify({"erro": "Não foi possível consultar as medições."}), 500

    distancia = float(est["distancia"][0])
    resposta = {
        "metrica": ctx["metrica"],
        "latitude": latitude,
        "longitude": longitude,
        "locais_com_medicao": ctx["locais"],
        "distancia_sensor_mais_proximo_m": round(distancia, 1),
        "raio_cobertura_m": RAIO_COBERTURA_M,
        "coberto_por_sensor": distancia <= RAIO_COBERTURA_M,
    }
    if "idw" in est:
        resposta["idw"] = {"valor_db": _r2(est["idw"][0]), "potencia": ctx["potencia"]}
    if "ml" in est:
        resposta["ml"] = {
            "valor_db": _r2(est["ml"][0]),
            "desvio_padrao_db": _r2(est["ml_desvio"][0]),
            **est["ml_parametros"],
        }
    elif ctx["metodo"] == "ambos":
        resposta["ml"] = {"erro": f"Requer ao menos {MIN_PONTOS_ML} locais distintos."}
    return jsonify(resposta), 200


@app.get("/api/ruido/grade")
def grade_ruido():
    """Mapa de ruído em grade regular (linhas = latitudes S→N, colunas = longitudes W→E)."""
    linhas = _arg_int("linhas", 20, 2, 100)
    colunas = _arg_int("colunas", 20, 2, 100)
    if linhas * colunas > MAX_CELULAS_GRADE:
        raise ErroDeRequisicao(f"A grade não pode ter mais de {MAX_CELULAS_GRADE} células.")

    ctx = _contexto()

    # Limites: informados pelo usuário ou o retângulo dos sensores + 10% de margem.
    def limites(nome_min, nome_max, serie, abs_max):
        lo, hi = float(serie.min()), float(serie.max())
        margem = max((hi - lo) * 0.1, 0.0005)
        vmin = _arg_float(nome_min, lo - margem, -abs_max, abs_max)
        vmax = _arg_float(nome_max, hi + margem, -abs_max, abs_max)
        if vmin >= vmax:
            raise ErroDeRequisicao(f"'{nome_min}' deve ser menor que '{nome_max}'.")
        return vmin, vmax

    lat_min, lat_max = limites("lat_min", "lat_max", ctx["lat"], 90)
    lon_min, lon_max = limites("lon_min", "lon_max", ctx["lon"], 180)

    lats = np.linspace(lat_min, lat_max, linhas)
    lons = np.linspace(lon_min, lon_max, colunas)
    malha_lon, malha_lat = np.meshgrid(lons, lats)

    try:
        est = _estimar(ctx, malha_lat.ravel(), malha_lon.ravel())
    except Error:
        app.logger.exception("Erro ao consultar medições no MySQL")
        return jsonify({"erro": "Não foi possível consultar as medições."}), 500

    def matriz(v, casas=2):
        return np.round(v.reshape(linhas, colunas), casas).tolist()

    resposta = {
        "metrica": ctx["metrica"],
        "limites": {"lat_min": lat_min, "lat_max": lat_max,
                    "lon_min": lon_min, "lon_max": lon_max},
        "linhas": linhas,
        "colunas": colunas,
        "latitudes": np.round(lats, 6).tolist(),
        "longitudes": np.round(lons, 6).tolist(),
        "locais_com_medicao": ctx["locais"],
        "raio_cobertura_m": RAIO_COBERTURA_M,
        "distancia_sensor_m": matriz(est["distancia"], 1),
        "coberta_por_sensor": (est["distancia"] <= RAIO_COBERTURA_M)
        .reshape(linhas, colunas).tolist(),
    }
    if "idw" in est:
        resposta["idw_db"] = matriz(est["idw"])
        resposta["idw_potencia"] = ctx["potencia"]
    if "ml" in est:
        resposta["ml_db"] = matriz(est["ml"])
        resposta["ml_desvio_padrao_db"] = matriz(est["ml_desvio"])
        resposta["ml_parametros"] = est["ml_parametros"]
    elif ctx["metodo"] == "ambos":
        resposta["ml_aviso"] = f"Modelo de ML requer ao menos {MIN_PONTOS_ML} locais distintos."
    return jsonify(resposta), 200


@app.get("/api/ruido/avaliacao")
def avaliar_modelos():
    """Validação cruzada leave-one-out: compara o erro do IDW com o do ML."""
    ctx = _contexto()
    if ctx["locais"] < 3:
        raise ErroDeRequisicao(
            "A validação cruzada precisa de ao menos 3 locais distintos.", 422)

    resposta = {
        "metrica": ctx["metrica"],
        "locais_com_medicao": ctx["locais"],
        "idw": {**metricas_erro(ctx["valores"], idw_validacao_cruzada(
            ctx["xy"], ctx["valores"], ctx["potencia"])), "potencia": ctx["potencia"]},
    }
    if ctx["ml_disponivel"]:
        modelo = ml_model.obter_modelo(
            ctx["metrica"], ctx["lat"], ctx["lon"], ctx["valores"])
        resposta["ml"] = {**ml_model.avaliar(modelo), **modelo.parametros()}
    else:
        resposta["ml"] = {"erro": f"Requer ao menos {MIN_PONTOS_ML} locais distintos."}
    return jsonify(resposta), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
