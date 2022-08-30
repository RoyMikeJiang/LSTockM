-- MySQL dump 10.13  Distrib 8.0.23, for Win64 (x86_64)
--
-- ------------------------------------------------------
-- Server version	8.0.26

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `account_info`
--

DROP TABLE IF EXISTS `account_info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `account_info` (
  `account_id` int unsigned NOT NULL AUTO_INCREMENT,
  `username` varchar(100) NOT NULL,
  `password` varchar(100) NOT NULL,
  `token` varchar(100) DEFAULT NULL,
  `token_time` datetime DEFAULT NULL,
  PRIMARY KEY (`username`),
  UNIQUE KEY `account_info_UN` (`account_id`)
) ENGINE=InnoDB AUTO_INCREMENT=10 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `current_data`
--

DROP TABLE IF EXISTS `current_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `current_data` (
  `stock_id` int unsigned NOT NULL,
  `data_time` datetime NOT NULL DEFAULT ((now() - interval 1 day)),
  `currentPrice` float DEFAULT NULL,
  `ebitdaMargins` float DEFAULT NULL,
  `operatingMargins` float DEFAULT NULL,
  `returnOnAssets` float DEFAULT NULL,
  `returnOnEquity` float DEFAULT NULL,
  `pegRatio` float DEFAULT NULL,
  `priceToBook` float DEFAULT NULL,
  `volume` bigint unsigned DEFAULT NULL,
  `bid` float DEFAULT NULL,
  `bidSize` bigint unsigned DEFAULT NULL,
  `ask` float DEFAULT NULL,
  `askSize` bigint unsigned DEFAULT NULL,
  `beta` float DEFAULT NULL,
  `quickRatio` float DEFAULT NULL,
  `valueAtRisk` float DEFAULT NULL,
  PRIMARY KEY (`stock_id`),
  CONSTRAINT `current_price_FK` FOREIGN KEY (`stock_id`) REFERENCES `stock_info` (`stock_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `data_source_info`
--

DROP TABLE IF EXISTS `data_source_info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `data_source_info` (
  `data_source_id` int unsigned NOT NULL AUTO_INCREMENT,
  `data_source_name` varchar(16) NOT NULL,
  `data_source_url` varchar(256) NOT NULL,
  PRIMARY KEY (`data_source_id`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `exchange_info`
--

DROP TABLE IF EXISTS `exchange_info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `exchange_info` (
  `exchange_id` int unsigned NOT NULL AUTO_INCREMENT,
  `exchange_name` varchar(16) NOT NULL,
  `currency` varchar(16) NOT NULL,
  PRIMARY KEY (`exchange_id`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `lstm_path`
--

DROP TABLE IF EXISTS `lstm_path`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `lstm_path` (
  `stock_id` int unsigned NOT NULL,
  `model_path` varchar(255) NOT NULL,
  PRIMARY KEY (`stock_id`),
  CONSTRAINT `lstm_path_FK` FOREIGN KEY (`stock_id`) REFERENCES `stock_info` (`stock_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ohlcv`
--

DROP TABLE IF EXISTS `ohlcv`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ohlcv` (
  `stock_id` int unsigned NOT NULL,
  `data_source_id` int unsigned NOT NULL,
  `price_date` date NOT NULL DEFAULT (curdate()),
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `open_price` float NOT NULL,
  `high_price` float NOT NULL,
  `low_price` float NOT NULL,
  `close_price` float NOT NULL,
  `adj_close_price` float NOT NULL,
  `volume` bigint unsigned NOT NULL,
  PRIMARY KEY (`stock_id`,`data_source_id`,`price_date`,`update_time`),
  KEY `ohlcv_FK_1` (`data_source_id`),
  CONSTRAINT `ohlcv_FK` FOREIGN KEY (`stock_id`) REFERENCES `stock_info` (`stock_id`),
  CONSTRAINT `ohlcv_FK_1` FOREIGN KEY (`data_source_id`) REFERENCES `data_source_info` (`data_source_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `predict_price`
--

DROP TABLE IF EXISTS `predict_price`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `predict_price` (
  `stock_id` int unsigned NOT NULL,
  `predict_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `price_date` date NOT NULL DEFAULT ((curdate() + interval 1 day)),
  `predict_price` float NOT NULL,
  `predict_id` int unsigned NOT NULL,
  PRIMARY KEY (`stock_id`,`predict_id`),
  CONSTRAINT `predict_price_FK` FOREIGN KEY (`stock_id`) REFERENCES `stock_info` (`stock_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `predict_request`
--

DROP TABLE IF EXISTS `predict_request`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `predict_request` (
  `account_id` int unsigned NOT NULL,
  `request_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `stock_symbol` varchar(100) NOT NULL,
  `stock_id` int unsigned DEFAULT NULL,
  `status` varchar(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL DEFAULT 'unstarted',
  `finish_time` datetime DEFAULT NULL,
  PRIMARY KEY (`account_id`,`request_time`,`stock_symbol`),
  UNIQUE KEY `predict_request_UN` (`stock_symbol`),
  KEY `predict_request_FK_1` (`stock_id`),
  CONSTRAINT `predict_request_FK` FOREIGN KEY (`account_id`) REFERENCES `account_info` (`account_id`),
  CONSTRAINT `predict_request_FK_1` FOREIGN KEY (`stock_id`) REFERENCES `stock_info` (`stock_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `recommendations`
--

DROP TABLE IF EXISTS `recommendations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `recommendations` (
  `stock_id` int unsigned NOT NULL,
  `rcmd_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `firm` varchar(32) NOT NULL,
  `to_grade` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `from_grade` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  `action` varchar(32) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci DEFAULT NULL,
  PRIMARY KEY (`stock_id`,`rcmd_time`,`firm`),
  CONSTRAINT `recommendations_FK` FOREIGN KEY (`stock_id`) REFERENCES `stock_info` (`stock_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `stock_info`
--

DROP TABLE IF EXISTS `stock_info`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `stock_info` (
  `stock_id` int unsigned NOT NULL AUTO_INCREMENT,
  `stock_symbol` varchar(32) NOT NULL,
  `company_name` varchar(128) NOT NULL,
  `exchange_id` int unsigned NOT NULL,
  PRIMARY KEY (`stock_id`),
  KEY `stock_info_FK` (`exchange_id`),
  CONSTRAINT `stock_info_FK` FOREIGN KEY (`exchange_id`) REFERENCES `exchange_info` (`exchange_id`)
) ENGINE=InnoDB AUTO_INCREMENT=28 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `watchlist`
--

DROP TABLE IF EXISTS `watchlist`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `watchlist` (
  `account_id` int unsigned NOT NULL,
  `stock_id` int unsigned NOT NULL,
  PRIMARY KEY (`account_id`,`stock_id`),
  KEY `watchlist_FK` (`stock_id`),
  CONSTRAINT `watchlist_FK` FOREIGN KEY (`stock_id`) REFERENCES `stock_info` (`stock_id`),
  CONSTRAINT `watchlist_FK_1` FOREIGN KEY (`account_id`) REFERENCES `account_info` (`account_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping routines for database 'lstockm'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2022-07-28 10:42:26
