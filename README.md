# API de medições acústicas (Flask + MySQL)

## 1. Preparar o banco

Execute o arquivo `schema.sql` no MySQL:

```bash
mysql -u root -p < schema.sql
```

Se você **já tem** o banco criado com a versão anterior, rode apenas a migração
(as medições antigas ficam com latitude/longitude `NULL` e são ignoradas na
interpolação):

```bash
mysql -u root -p < migrations/001_add_latitude_longitude.sql
```

## 2. Instalar e configurar

```bash
python -m venv .venv
```

Ative o ambiente virtual:

- Linux/macOS: `source .venv/bin/activate`
- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`

Instale as dependências e configure as variáveis de ambiente:

```bash
pip install -r requirements.txt
```

Copie `.env.example` para `.env` e informe as credenciais do MySQL. O arquivo `.env`
será carregado automaticamente pela aplicação.

## 3. Executar

```bash
python app.py
```

A API ficará disponível em `http://localhost:5000`.

## 4. Endpoints

- `GET /health` — verifica se a aplicação está no ar.
- `POST /api/medicoes` — grava uma medição.
- `GET /api/medicoes` — lista as medições.
- `GET /api/medicoes/<id>` — consulta uma medição pelo ID.
- `GET /api/ruido/estimativa` — estima o ruído em uma coordenada (IDW e/ou ML).
- `GET /api/ruido/grade` — mapa de ruído em grade regular.
- `GET /api/ruido/avaliacao` — compara o erro do IDW com o do ML (validação cruzada).

### Exemplo de envio

```bash
curl -X POST http://localhost:5000/api/medicoes \
  -H "Content-Type: application/json" \
  -d '{
    "ponderacao_frequencia": "A",
    "tempo_resposta": "Rápido (F - 200ms)",
    "calibracao": "Linear",
    "avg_db": 81.2,
    "min_db": 69.3,
    "max_db": 92.7,
    "peak_db": 96.2,
    "latitude": -23.550520,
    "longitude": -46.633308
  }'
```

Resposta esperada (`201 Created`):

```json
{
  "mensagem": "Medição armazenada com sucesso.",
  "id": 1
}
```

`latitude` (-90 a 90) e `longitude` (-180 a 180) são obrigatórias e gravadas como
`DECIMAL(9,6)` (cerca de 0,11 m de resolução).

Os campos numéricos são armazenados como `DECIMAL(6,2)` para preservar duas casas
decimais. A API também valida a consistência básica dos valores: mínimo <= média <=
máximo <= pico.

## 5. Interpolação espacial e estimativa por ML

Com as coordenadas, a API estima o ruído em locais **sem sensor** de duas formas:

1. **IDW (inverso da distância ponderada)** — interpolação determinística. Cada ponto
   recebe a média dos sensores ponderada por `1/distância^potência` (padrão 2). Nos
   sensores devolve exatamente o valor medido, e nunca passa do intervalo medido.
2. **Regressão por Processo Gaussiano (scikit-learn)** — modelo de ML equivalente a
   um kriging com parâmetros aprendidos dos dados (escala de correlação espacial,
   variância e ruído de medição). Além da estimativa devolve o **desvio-padrão**, que
   cresce longe dos sensores e indica a confiança.

As coordenadas são projetadas para metros (projeção equirretangular local), então as
distâncias são reais. Medições no mesmo local são agrupadas (média) antes do cálculo.
O modelo é treinado sob demanda e mantido em cache até os dados mudarem.

Parâmetros comuns: `metrica` (`avg_db` padrão, `min_db`, `max_db`, `peak_db`),
`metodo` (`idw`, `ml` ou `ambos` — padrão) e `potencia` do IDW (0,5 a 6; padrão 2).

### Estimativa em um ponto

```bash
curl "http://localhost:5000/api/ruido/estimativa?latitude=-23.5510&longitude=-46.6340"
```

```json
{
  "metrica": "avg_db",
  "latitude": -23.551,
  "longitude": -46.634,
  "locais_com_medicao": 12,
  "distancia_sensor_mais_proximo_m": 38.2,
  "raio_cobertura_m": 50.0,
  "coberto_por_sensor": true,
  "idw": {"valor_db": 78.41, "potencia": 2.0},
  "ml": {"valor_db": 77.96, "desvio_padrao_db": 1.12, "modelo": "gaussian_process",
         "escala_correlacao_m": 142.3, "pontos_treino": 12}
}
```

(Valores ilustrativos.)

### Mapa em grade

```bash
curl "http://localhost:5000/api/ruido/grade?linhas=30&colunas=30&metodo=ambos"
```

Parâmetros opcionais: `linhas`/`colunas` (2 a 100, padrão 20), `lat_min`, `lat_max`,
`lon_min`, `lon_max` (padrão: retângulo dos sensores + 10 % de margem). Retorna
matrizes `[linha][coluna]`, com linhas = `latitudes` (sul→norte) e colunas =
`longitudes` (oeste→leste): `idw_db`, `ml_db`, `ml_desvio_padrao_db`,
`distancia_sensor_m` e `coberta_por_sensor` (células dentro de `RAIO_COBERTURA_M` de
algum sensor; as demais são as áreas **estimadas**).

### Avaliação dos modelos

```bash
curl "http://localhost:5000/api/ruido/avaliacao"
```

Validação cruzada *leave-one-out*: cada sensor é previsto usando só os outros, e a
API devolve RMSE, MAE e viés em dB do IDW e do ML. Use para decidir qual método
confiar com os seus dados.

### Limitações importantes

- O ML precisa de ao menos `MIN_PONTOS_ML` locais distintos (padrão 5); com menos,
  `metodo=ml` responde `422` e `metodo=ambos` devolve só o IDW. Na prática, mais
  pontos e boa distribuição espacial importam mais que o algoritmo.
- Ambos os métodos usam **somente a posição**. Eles não conhecem ruas, edifícios,
  fontes sonoras nem barreiras; em áreas muito distantes dos sensores a estimativa
  tende à média regional (o desvio-padrão do ML sinaliza isso). Trate as áreas
  fora de `coberto_por_sensor` como estimativas, não medições.
- A projeção local é adequada para áreas de até algumas dezenas de km.
- A média em dB é aritmética sobre os valores armazenados (como já era no projeto).
  Para rigor acústico, a média energética seria `10·log10(mean(10^(L/10)))`.

## 6. Testes

Os testes usam dados sintéticos e não precisam de MySQL:

```bash
python -m unittest discover -s tests -v
```
