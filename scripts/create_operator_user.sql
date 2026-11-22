-- ==============================================================================
-- StudyTrade — Operator User Creation & Activation SQL
-- Target Database: web/db.sqlite3
-- Operator Email: tusharpgtbiology@gmail.com
-- Username:       tushar
-- Password:       StudyTrade@2026
-- ==============================================================================

INSERT INTO auth_user (
    id,
    password,
    last_login,
    is_superuser,
    username,
    first_name,
    last_name,
    email,
    is_staff,
    is_active,
    date_joined
) VALUES (
    1,
    'pbkdf2_sha256$1500000$F7NgGj2yeeRT5Ncga5GVSG$8o4Up37OVV6FUpg860+iTE+emQJNL8JepmVrtub46sQ=',
    datetime('now'),
    1,
    'tushar',
    'Tushar',
    'Operator',
    'tusharpgtbiology@gmail.com',
    1,
    1,
    datetime('now')
)
ON CONFLICT(id) DO UPDATE SET
    username = 'tushar',
    email = 'tusharpgtbiology@gmail.com',
    password = 'pbkdf2_sha256$1500000$F7NgGj2yeeRT5Ncga5GVSG$8o4Up37OVV6FUpg860+iTE+emQJNL8JepmVrtub46sQ=',
    is_superuser = 1,
    is_staff = 1,
    is_active = 1;
