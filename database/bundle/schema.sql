CREATE TABLE alembic_version (
	version_num VARCHAR(32) NOT NULL, 
	CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);
CREATE TABLE hardware_versions (
	id VARCHAR(100) NOT NULL, 
	payload JSON NOT NULL, 
	PRIMARY KEY (id)
);
CREATE TABLE model_structure_snapshots (
	id VARCHAR(64) NOT NULL, 
	model_id VARCHAR(100) NOT NULL, 
	payload JSON NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(model_id) REFERENCES model_versions (id)
);
CREATE TABLE model_versions (
	id VARCHAR(100) NOT NULL, 
	repo_id VARCHAR(200) NOT NULL, 
	revision VARCHAR(40) NOT NULL, 
	payload JSON NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (repo_id, revision)
);
CREATE TABLE workload_design_revisions (
	id VARCHAR(64) NOT NULL, 
	design_id VARCHAR(64) NOT NULL, 
	revision_no INTEGER NOT NULL, 
	model_id VARCHAR(100) NOT NULL, 
	hardware_id VARCHAR(100) NOT NULL, 
	payload JSON NOT NULL, 
	input_hash VARCHAR(64) NOT NULL, 
	schema_version VARCHAR(50) NOT NULL, 
	created_at VARCHAR(40) NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (design_id, revision_no), 
	FOREIGN KEY(design_id) REFERENCES workload_designs (id) ON DELETE CASCADE, 
	FOREIGN KEY(model_id) REFERENCES model_versions (id) ON DELETE RESTRICT, 
	FOREIGN KEY(hardware_id) REFERENCES hardware_versions (id) ON DELETE RESTRICT
);
CREATE TABLE workload_designs (
	id VARCHAR(64) NOT NULL, 
	name VARCHAR(100) NOT NULL, 
	head_revision INTEGER NOT NULL, 
	created_at VARCHAR(40) NOT NULL, 
	updated_at VARCHAR(40) NOT NULL, 
	idempotency_key VARCHAR(200) NOT NULL, 
	request_hash VARCHAR(64) NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (idempotency_key)
);
CREATE TABLE workload_plan_details (
	plan_id VARCHAR(64) NOT NULL, 
	kind VARCHAR(30) NOT NULL, 
	chunk_key INTEGER NOT NULL, 
	payload JSON NOT NULL, 
	hash VARCHAR(64) NOT NULL, 
	PRIMARY KEY (plan_id, kind, chunk_key), 
	FOREIGN KEY(plan_id) REFERENCES workload_plans (id) ON DELETE CASCADE
);
CREATE TABLE workload_plans (
	id VARCHAR(64) NOT NULL, 
	run_id VARCHAR(64) NOT NULL, 
	candidate_key VARCHAR(200) NOT NULL, 
	rank INTEGER NOT NULL, 
	summary JSON NOT NULL, 
	result_hash VARCHAR(64) NOT NULL, 
	PRIMARY KEY (id), 
	UNIQUE (run_id, candidate_key), 
	FOREIGN KEY(run_id) REFERENCES workload_runs (id) ON DELETE CASCADE
);
CREATE TABLE workload_runs (
	id VARCHAR(64) NOT NULL, 
	revision_id VARCHAR(64) NOT NULL, 
	structure_snapshot_id VARCHAR(64) NOT NULL, 
	status VARCHAR(30) NOT NULL, 
	idempotency_key VARCHAR(200) NOT NULL, 
	request_hash VARCHAR(64) NOT NULL, 
	input_hash VARCHAR(64) NOT NULL, 
	frozen_input JSON NOT NULL, 
	versions JSON NOT NULL, 
	queued_at VARCHAR(40) NOT NULL, 
	started_at VARCHAR(40), 
	finished_at VARCHAR(40), 
	lease_expires_at VARCHAR(40), 
	owner VARCHAR(64), 
	attempt INTEGER NOT NULL, 
	progress JSON NOT NULL, 
	error JSON, 
	result_manifest JSON, 
	PRIMARY KEY (id), 
	UNIQUE (idempotency_key), 
	FOREIGN KEY(revision_id) REFERENCES workload_design_revisions (id) ON DELETE CASCADE, 
	FOREIGN KEY(structure_snapshot_id) REFERENCES model_structure_snapshots (id) ON DELETE RESTRICT
);
CREATE INDEX ix_model_structure_snapshots_model_id ON model_structure_snapshots (model_id);
CREATE INDEX ix_workload_design_revisions_design_id ON workload_design_revisions (design_id);
CREATE INDEX ix_workload_designs_created_at ON workload_designs (created_at);
CREATE INDEX ix_workload_plans_rank ON workload_plans (rank);
CREATE INDEX ix_workload_plans_run_id ON workload_plans (run_id);
CREATE INDEX ix_workload_runs_queued_at ON workload_runs (queued_at);
CREATE INDEX ix_workload_runs_revision_id ON workload_runs (revision_id);
CREATE INDEX ix_workload_runs_status ON workload_runs (status);
