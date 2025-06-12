-- Tạo database nếu chưa tồn tại
CREATE DATABASE IF NOT EXISTS loan_scoring_db;
USE loan_scoring_db;

-- Tạo bảng loan_applications
CREATE TABLE IF NOT EXISTS loan_applications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    annual_income FLOAT NOT NULL,
    credit_score INT NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(10) NOT NULL,
    num_of_credit_accounts INT NOT NULL,
    num_of_late_payments INT NOT NULL,
    debt_to_income_ratio FLOAT NOT NULL,
    region VARCHAR(50) NOT NULL,
    label INT NOT NULL
); 