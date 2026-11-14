-- Para bancos JÁ EXISTENTES (criados com a versão anterior do schema.sql).
-- Em instalações novas, basta executar o schema.sql.
USE medicoes_db;

-- As colunas ficam NULL nas medições antigas (não há coordenadas para elas).
-- A API exige latitude/longitude em novas medições e ignora registros sem
-- coordenadas na interpolação e no modelo de ML.
ALTER TABLE medicoes
    ADD COLUMN latitude DECIMAL(9,6) NULL AFTER peak_db,
    ADD COLUMN longitude DECIMAL(9,6) NULL AFTER latitude,
    ADD INDEX idx_medicoes_lat_lon (latitude, longitude);

-- Opcional: se você preencher as coordenadas das medições antigas, pode
-- tornar as colunas obrigatórias:
-- ALTER TABLE medicoes
--     MODIFY latitude DECIMAL(9,6) NOT NULL,
--     MODIFY longitude DECIMAL(9,6) NOT NULL;
