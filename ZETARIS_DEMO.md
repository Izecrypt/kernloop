# Zetaris Demo Challenge plan (add-on prize, separate from the Nebius x NVIDIA submission)

Status date: 2026-10-28. Video due Oct 29, 23:45 UTC. Requirement: short demo
of the build with Zetaris working live, including 40 to 60 seconds on why it
is amazing. Honesty rule: the video must show Zetaris genuinely querying real
KernLoop run-log data from /Users/tbo/kernloop/runs, never mocked data.

## 1. Access status

Blocked at the Zetaris Cloud login, and only at that step.

- Login URL: https://cloud.enterprise.zetaris.com/login
- Account: tobiobembeofficial@gmail.com (auto-provisioned by the hackathon).
- The login form is email only; the API flow is `auth/login/start` then
  `auth/login/otp/send` and `auth/login/otp/verify`. The OTP is emailed to
  the Gmail above. No password option is offered for this account.
- I probed the API directly. Every unauthenticated endpoint
  (`/api/auth/methods`, `/api/auth/login/start`) returns 401 `Unauthorized`
  even with full browser headers, so the OTP email cannot be triggered
  from here. A browser session would face the same email gate.
- The local Docker path is confirmed real and documented (Zetaris blog and
  the getting-started repo): after login, the console shows registry
  credentials, and a "Run locally" flow yields `zetaris-platform.zip` with
  `preflight.sh`, `docker compose`, and `verify.sh`. The old public
  `zetaris/Freemium` compose file references images
  (`zetaris/lightning-server:v2.4.1.0-latest`) whose registry now requires
  the cloud login, so there is no way around the OTP.

What the owner must do (about 10 minutes):

1. Open https://cloud.enterprise.zetaris.com/login, enter
   tobiobembeofficial@gmail.com, and submit.
2. Open the Gmail inbox, copy the 6 digit OTP from the Zetaris email, paste
   it, and complete login.
3. If the console offers "Mini template cluster" versus "Run locally",
   choose **Run locally** (no AWS account needed).
4. On the local setup screen, copy the two registry credential lines and
   paste them into a Terminal here (they run `docker login ...`; expect
   "Login Succeeded").
5. Download `zetaris-platform.zip` and move it to /Users/tbo (macOS blocks
   Terminal reads of ~/Downloads).
6. Tell me it is done (or run: `cd /Users/tbo && [ -d zetaris-platform ] ||
   unzip -q zetaris-platform.zip && cd zetaris-platform && ./preflight.sh
   && docker compose pull && docker compose up -d && ./verify.sh`).

Alternative if the owner prefers to finish by hand: after step 2, in the
console choose "Run locally", follow the on-screen steps (Docker Desktop
8 GB RAM and 4 CPUs, registry login, unzip, preflight, compose up), then
confirm the gateway answers at http://localhost:3000. I take over from
there: staging the run logs, registering the source, and rehearsing queries.

If the OTP email never arrives, the account may need activation from a
provisioning email in that same inbox; click its activation link first.

## 2. Connector answer (design question)

Zetaris can register the run-log directory as a queryable source, and the
right connector is the **Lightning Filestore** with **FORMAT JSON**, no
JDBC or REST shim needed:

```sql
CREATE LIGHTNING DATABASE kernloop;
CREATE LIGHTNING FILESTORE TABLE iterations FROM kernloop FORMAT JSON OPTIONS (
  path "/home/zetaris/data/shared/kernloop-runs/*.runlog.jsonl"
);
```

- `CREATE LIGHTNING FILESTORE TABLE ... FORMAT JSON` registers files or
  directories with schema inferred from the data (confirmed in Zetaris
  getting-started docs, Lightning SQL commands, section 5.3). JSON, CSV,
  Parquet, and more are supported; Spark JSON reads line-delimited records,
  which matches JSONL.
- Local files must live under the container shared root (the compose file
  bind-mounts `customer_data` at `/home/zetaris/data`), so tomorrow's wiring
  either mounts /Users/tbo/kernloop/runs there or copies `*.runlog.jsonl`
  into the shared dir (copy of real files only, no transformation, so the
  honesty rule holds).
- The runs directory also contains `.html` viewer files; the path glob
  `*.runlog.jsonl` excludes them.
- Alternative connectors if the file path fights us: S3 via
  `path "s3a://..."` with `awsAccessKeyId`/`awsSecretAccessKey` options, or
  `CREATE FILESTORE SESSION TABLE` for one-shot ad-hoc queries.
- Known wrinkle: the log mixes record shapes (run header lines, iteration
  records, research records). Spark unions the inferred schema, so non-
  iteration lines get nulls; queries filter `WHERE iteration IS NOT NULL`.

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
