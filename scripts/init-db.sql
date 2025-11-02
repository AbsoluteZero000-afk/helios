-- Helios v3 Database Initialization Script
-- Creates initial database structure and seed data

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create helios user if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_user WHERE usename = 'helios') THEN
        CREATE USER helios WITH PASSWORD 'helios';
    END IF;
END
$$;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE helios TO helios;
ALTER USER helios CREATEDB;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS system;

-- Grant schema privileges
GRANT ALL PRIVILEGES ON SCHEMA trading TO helios;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO helios;
GRANT ALL PRIVILEGES ON SCHEMA system TO helios;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA trading GRANT ALL ON TABLES TO helios;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT ALL ON TABLES TO helios;
ALTER DEFAULT PRIVILEGES IN SCHEMA system GRANT ALL ON TABLES TO helios;

-- Create indexes for performance
-- Note: Specific table indexes will be created by SQLAlchemy migrations

-- Insert seed data if needed
-- This could include initial strategy configurations, risk parameters, etc.

-- Log initialization
INSERT INTO system.initialization_log (timestamp, component, status, details) 
VALUES (NOW(), 'database', 'completed', 'Initial database setup completed')
ON CONFLICT DO NOTHING;