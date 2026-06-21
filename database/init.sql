CREATE DATABASE IF NOT EXISTS medical_ai DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE medical_ai;

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    hashed_password TEXT NOT NULL,
    real_name VARCHAR(50) DEFAULT '',
    role ENUM('doctor', 'admin') NOT NULL DEFAULT 'doctor',
    department VARCHAR(100) DEFAULT '',
    avatar VARCHAR(255) DEFAULT '',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 虚拟患者表
CREATE TABLE IF NOT EXISTS virtual_patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    age INT NOT NULL,
    gender ENUM('male', 'female') NOT NULL,
    personality_type ENUM('配合型', '焦虑型', '沉默型', '对抗型') NOT NULL COMMENT '人格类型',
    chief_complaint VARCHAR(200) NOT NULL COMMENT '主诉',
    medical_history TEXT NOT NULL COMMENT '病史',
    symptoms TEXT NOT NULL COMMENT '症状描述（JSON）',
    expected_diagnosis VARCHAR(200) DEFAULT '' COMMENT '预期诊断',
    system_prompt TEXT NOT NULL COMMENT '虚拟患者系统提示词',
    difficulty_level INT DEFAULT 1 COMMENT '难度等级 1-5',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 问诊会话表
CREATE TABLE IF NOT EXISTS consultations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    doctor_id INT NOT NULL,
    patient_id INT NOT NULL,
    status ENUM('in_progress', 'completed', 'evaluated') DEFAULT 'in_progress',
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ended_at DATETIME DEFAULT NULL,
    summary TEXT DEFAULT NULL,
    diagnosis TEXT DEFAULT NULL COMMENT '医生提交的诊断结果',
    treatment_plan TEXT DEFAULT NULL COMMENT '医生提交的治疗方案',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_doctor_id (doctor_id),
    FOREIGN KEY (doctor_id) REFERENCES users(id),
    FOREIGN KEY (patient_id) REFERENCES virtual_patients(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 问诊消息表
CREATE TABLE IF NOT EXISTS consultation_messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    consultation_id INT NOT NULL,
    role ENUM('doctor', 'patient') NOT NULL,
    content TEXT NOT NULL,
    sequence INT NOT NULL COMMENT '消息序号',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_consultation_id (consultation_id),
    FOREIGN KEY (consultation_id) REFERENCES consultations(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 评估报告表（五维度评估）
CREATE TABLE IF NOT EXISTS evaluations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    consultation_id INT NOT NULL UNIQUE,
    inquiry_score FLOAT DEFAULT 0 COMMENT '病史采集评分',
    inquiry_analysis TEXT COMMENT '病史采集分析',
    knowledge_score FLOAT NULL DEFAULT NULL COMMENT '医学知识评分',
    knowledge_analysis TEXT COMMENT '知识核对分析',
    humanistic_score FLOAT DEFAULT 0 COMMENT '沟通交流评分',
    humanistic_analysis TEXT COMMENT '沟通交流分析',
    diagnosis_score FLOAT DEFAULT 0 COMMENT '诊断结果评分',
    diagnosis_analysis TEXT COMMENT '诊断结果分析',
    treatment_score FLOAT DEFAULT 0 COMMENT '治疗方案评分',
    treatment_analysis TEXT COMMENT '治疗方案分析',
    total_score FLOAT NULL DEFAULT NULL COMMENT '综合评分',
    overall_summary TEXT COMMENT '综合评估摘要',
    improvement_suggestions TEXT COMMENT '改进建议',
    citation_data JSON NULL COMMENT '引用数据',
    retrieval_status VARCHAR(20) NOT NULL DEFAULT 'not_run' COMMENT '检索状态',
    evidence_stance VARCHAR(20) NOT NULL DEFAULT 'undetermined' COMMENT '证据立场',
    human_review_needed BOOLEAN NOT NULL DEFAULT FALSE COMMENT '是否需要人工复核',
    review_reason TEXT NULL COMMENT '复核原因',
    rag_trace_data JSON NULL COMMENT 'RAG追踪数据',
    evaluation_status VARCHAR(20) NOT NULL DEFAULT 'completed' COMMENT '评估状态',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_consultation_id (consultation_id),
    FOREIGN KEY (consultation_id) REFERENCES consultations(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
