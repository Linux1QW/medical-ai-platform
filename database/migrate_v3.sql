USE medical_ai;

ALTER TABLE users
    MODIFY COLUMN hashed_password TEXT NOT NULL;
