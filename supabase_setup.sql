-- Supabase Database Schema for Family Tree
-- Run this SQL in your Supabase SQL Editor

-- Create people table
CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    gender TEXT CHECK (gender IN ('male', 'female')),
    image_path TEXT,
    birth_year TEXT,
    death_year TEXT,
    birth_order INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create edges table (parent-child relationships)
CREATE TABLE IF NOT EXISTS edges (
    id SERIAL PRIMARY KEY,
    parent_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    child_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(parent_id, child_id)
);

-- Create spouses table
CREATE TABLE IF NOT EXISTS spouses (
    id SERIAL PRIMARY KEY,
    person1_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    person2_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(person1_id, person2_id)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges(parent_id);
CREATE INDEX IF NOT EXISTS idx_edges_child ON edges(child_id);
CREATE INDEX IF NOT EXISTS idx_spouses_person1 ON spouses(person1_id);
CREATE INDEX IF NOT EXISTS idx_spouses_person2 ON spouses(person2_id);

-- Enable Row Level Security (optional, for public access)
ALTER TABLE people ENABLE ROW LEVEL SECURITY;
ALTER TABLE edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE spouses ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (anyone can read/write)
-- Remove these if you want to restrict access
CREATE POLICY "Allow public read access on people" ON people FOR SELECT USING (true);
CREATE POLICY "Allow public insert access on people" ON people FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public update access on people" ON people FOR UPDATE USING (true);
CREATE POLICY "Allow public delete access on people" ON people FOR DELETE USING (true);

CREATE POLICY "Allow public read access on edges" ON edges FOR SELECT USING (true);
CREATE POLICY "Allow public insert access on edges" ON edges FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public delete access on edges" ON edges FOR DELETE USING (true);

CREATE POLICY "Allow public read access on spouses" ON spouses FOR SELECT USING (true);
CREATE POLICY "Allow public insert access on spouses" ON spouses FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow public delete access on spouses" ON spouses FOR DELETE USING (true);
