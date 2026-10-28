# Zetaris Demo Challenge plan (add-on prize, separate from the Nebius x NVIDIA submission)

Status date: 2026-10-28 (evening): LOCAL STACK LIVE, ALL FIVE QUERIES REHEARSED.
Video due Oct 29, 23:45 UTC. Requirement: short demo of the build with Zetaris
working live, including 40 to 60 seconds on why it is amazing. Honesty rule:
the video shows Zetaris genuinely querying real KernLoop run-log data from
/Users/tbo/kernloop/runs, never mocked data.

## 1. Access status: UNBLOCKED (local docker path, owner completed 2026-10-28)

- Platform running at http://localhost:3000 (login admin@zetaris.com, password
  in /Users/tbo/zetaris-platform/.env as ZETARIS_ADMIN_PASSWORD).
- Installed at /Users/tbo/zetaris-platform (registry login done; images pulled;
  verify.sh passes except bootstrap sample-data load, which is irrelevant).
- Docker note: Docker.raw had ballooned to 228GB on a full disk and corrupted
  itself; deleted and recreated (owner approved; w3life-db dev container lost,
  logged in vault). Rosetta confirmed working (preflight warning is cosmetic).

### Working SQL execution path (headless, same path the UI uses)

- Login: POST http://localhost:3000/api/auth/login {username, password} ->
  response carries idToken and orgId.
- Run any SQL (DDL + SELECT): POST http://localhost:3000/api/proxy/sql-editor/sqls/run
  with headers Authorization: Bearer <idToken>, X-Org-ID: <orgId>,
  X-Request-ID: <uuid>, body {"queryId": <fresh uuid>, "sql": "...", "source": "SqlEditor",
  "chunkSize": 10000}. Returns {"headers", "data", "total", "timeUsed"}.
- Helper script used for rehearsal: /tmp/zsql.py (recreate if gone).
- The /api/proxy/query/sql/start endpoint is SELECT-only; DDL must go through
  the sql-editor endpoint above.

### Filestore quirk discovered during rehearsal (important)

- Lightning Filestore FORMAT JSON reads exactly ONE record per file (it treats
  each file as a single JSON document). Multi-line JSONL files yield only the
  first record. Options multiLine / directory paths / globs do not change this.
- Working setup: the 21 iteration records were copied VERBATIM (unmodified
  rows, filtered subset) into one file per record:
  host /Users/tbo/kernloop/runs-iters-flat/*.json ->
  container /home/zetaris/customer-data/shared/kernloop-flat (docker compose cp).
- Setup SQL (live on camera):
  CREATE LIGHTNING DATABASE kernloop;
  CREATE LIGHTNING FILESTORE TABLE iterations FROM kernloop FORMAT JSON OPTIONS
    (path '/home/zetaris/customer-data/shared/kernloop-flat');
  SELECT COUNT(*) FROM kernloop.iterations;  -- expect 21
- Honesty note for the video: state that rows are verbatim copies of the real
  run logs, staged one record per file because of how the filestore reads JSON.

## 2. Connector: Lightning Filestore FORMAT JSON (confirmed working)

As designed below, with the quirk above. Original design notes follow.

- `CREATE LIGHTNING FILESTORE TABLE ... FORMAT JSON` registers files with
  schema inferred from data (Zetaris getting-started docs 5.3).
- Local files must live under the container shared root (bind-mounted
  customer-data volume at /home/zetaris/customer-data); driver env sets
  ZETARIS_LOCAL_FILE_SYTEM_ROOT_PATH=/home/zetaris/customer-data/shared.

## 3. Demo story (SQL over real run logs)

Setup (live on camera, typed once):

```sql
CREATE LIGHTNING DATABASE kernloop;
CREATE LIGHTNING FILESTORE TABLE iterations FROM kernloop FORMAT JSON OPTIONS (
  path "/home/zetaris/data/shared/kernloop-runs/*.runlog.jsonl"
);
SELECT COUNT(*) FROM kernloop.iterations WHERE iteration IS NOT NULL;
```

Q1, promotion trajectory: the champion line stepping down.

```sql
SELECT run_id, iteration, timing.mean_ms, decision.promoted
FROM kernloop.iterations
WHERE timing.mean_ms IS NOT NULL
ORDER BY run_id, iteration;
```

Q2, rejection-reason clustering: what actually blocked the agent.

```sql
SELECT decision.reason, COUNT(*) AS times
FROM kernloop.iterations
WHERE decision.promoted = false
GROUP BY decision.reason
ORDER BY times DESC;
```

Q3, tokens per speedup (cost of the win): per promotion, model tokens spent
against the improvement reason.

```sql
SELECT run_id, iteration,
       tokens.prompt + tokens.completion AS tokens_spent,
       timing.mean_ms, decision.reason
FROM kernloop.iterations
WHERE decision.promoted = true
ORDER BY run_id, iteration;
```

Q4, honest-log verification (the trust query): which backend produced which
number, and how much of the log is synthetic.

```sql
SELECT backend, COUNT(*) AS iterations,
       SUM(CASE WHEN timing.meta.synthetic = true THEN 1 ELSE 0 END) AS synthetic_rows
FROM kernloop.iterations
WHERE timing.mean_ms IS NOT NULL
GROUP BY backend;
```

Q5, best champion per run, with the model that wrote it.

```sql
SELECT run_id, MIN(timing.mean_ms) AS best_mean_ms, proposal.model
FROM kernloop.iterations
WHERE decision.promoted = true
GROUP BY run_id, proposal.model
ORDER BY best_mean_ms;
```

Why-it-is-amazing segment (40 to 60 seconds, spoken over Q2 plus Q4):
KernLoop's run log is an agent's complete decision audit trail, and Zetaris
turns it into a queryable system of record. Not a dashboard, not a parser:
real SQL over files, federated. You can ask why the agent failed (reason
clustering), what the improvements cost (tokens per promotion), and whether
its numbers are real (synthetic rows flagged per backend). That last query
is the pitch: an agent you can audit with a JOIN, not a promise. Judges can
type the queries themselves.

## 4. Plan for tomorrow (Oct 28 to 29)

1. Owner completes section 1 (login plus local install). If only the OTP is
   missing, the owner can paste the 6 digit code to me and I drive the REST
   login and download from there.
2. Stage `*.runlog.jsonl` into the container shared root; run the setup SQL;
   fix glob or schema wrinkles (Q1 through Q5 rehearsed; adjust field names
   if Spark infers differently).
3. Screen-record: terminal staging, the setup SQL, Q1 through Q5 in order,
   then the why-it-is-amazing close. Keep raw segments; assemble to under
   the prize duration rules. Publish and submit before 23:45 UTC.

Not in scope tonight: recording, editing, any change to the Nebius x NVIDIA
submission, and any repo commit beyond this plan document.
