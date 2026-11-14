CREATE DATABASE IF NOT EXISTS medicoes_db
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE medicoes_db;

CREATE TABLE IF NOT EXISTS medicoes (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    ponderacao_frequencia VARCHAR(20) NOT NULL,
    tempo_resposta VARCHAR(100) NOT NULL,
    calibracao VARCHAR(50) NOT NULL,
    avg_db DECIMAL(6,2) NOT NULL,
    min_db DECIMAL(6,2) NOT NULL,
    max_db DECIMAL(6,2) NOT NULL,
    peak_db DECIMAL(6,2) NOT NULL,
    latitude DECIMAL(9,6) NOT NULL,
    longitude DECIMAL(9,6) NOT NULL,
    criado_em TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    INDEX idx_medicoes_lat_lon (latitude, longitude)
);
