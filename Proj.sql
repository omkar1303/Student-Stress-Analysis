CREATE DATABASE student_stress_db;
USE student_stress_db;

CREATE TABLE IF NOT EXISTS feedback_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    certification_course INT,
    gender INT,
    department INT,
    stress_level INT,
    feedback TEXT
);

LOAD DATA INFILE
'C://Users//OMKAR PAWAR//Desktop//human_feedback.csv'
INTO TABLE feedback_data
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(certification_course, gender, department, stress_level, feedback);

SHOW VARIABLES LIKE 'secure_file_priv';
