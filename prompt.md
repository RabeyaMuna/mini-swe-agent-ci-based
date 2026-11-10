Create a publication-quality end-to-end methodology overview figure for an ICLR paper:

“ReCAP: Learning Problem-Level Repair Experience from Entangled Pull-Request Histories”

IMPORTANT:
Follow the methodology and terminology exactly.

Do NOT invent new major methodology stages or rename the components.

The figure should explain ReCAP through ONE continuous running example, from a completed historical CI repair to experience construction and finally reuse for a new CI failure.

The target audience is AI/ML researchers who may not be familiar with Continuous Integration (CI). Therefore, the figure must explain enough CI structure to make the ReCAP methodology understandable, but CI explanation should remain supporting context rather than dominate the figure.

============================================================
OVERALL FIGURE STORY
============================================================

Use a wide horizontal left-to-right layout.

Organize the figure into THREE major methodology stages:

1. MULTI-VIEW RECONSTRUCTION
   Learn problem-level repair records from completed historical CI repair histories.

2. REPAIR EXPERIENCE AT THREE LEVELS
   Represent each reconstructed problem-repair record at L1, L2, and L3.

3. DYNAMIC EXPERIENCE REUSE
   Retrieve and use relevant problem-level historical experience to repair a new CI failure.

These correspond to the three main stages of the ReCAP methodology.

Use ONE simple running example consistently throughout the figure.

The high-level visual flow should be:

HISTORICAL COMPLETED CI REPAIR
          ↓
MULTI-VIEW RECONSTRUCTION
          ↓
PROBLEM-LEVEL REPAIR RECORDS
          ↓
L1 / L2 / L3 EXPERIENCE
          ↓
NEW TARGET CI FAILURE
          ↓
DYNAMIC EXPERIENCE REUSE
          ↓
SEQUENTIAL REPAIR
          ↓
FINAL FULL CI PASS


################################################################
PART 0: MINIMAL CI CONTEXT FOR AN AI/ML AUDIENCE
################################################################

Before Stage 1, introduce the historical CI repair using a compact visual example.

Do NOT make “CI Concepts” a separate large methodology stage.

Instead, place a small explanatory panel at the beginning titled:

HISTORICAL CI REPAIR

Show:

Developer Changes
       ↓
Pull Request (PR)
       ↓
CI Workflow Triggered
       ↓
Automated Validation

Add a tiny explanatory sentence:

“CI automatically validates proposed code changes.”

------------------------------------------------------------
CI WORKFLOW STRUCTURE
------------------------------------------------------------

Show a compact CI workflow.

Label:

CI WORKFLOW
“What should execute”

Make the hierarchy understandable:

Workflow
  ↓
Jobs
  ↓
Steps
  ↓
Commands

Use this running example:

CI Workflow

Job 1: Setup
  Step: Install dependencies
  $ pip install -r requirements.txt

Job 2: Unit Tests
  Step: Run tests
  $ pytest tests/

Job 3: Integration Tests
  Step: Run integration tests
  $ pytest tests/integration/

Show dependencies:

Setup
  ↓
Unit Tests
  ↓
Integration Tests

Keep this small and visual.

------------------------------------------------------------
CI EXECUTION AND LOG
------------------------------------------------------------

Next to the workflow, show the corresponding execution.

Label:

CI EXECUTION
“What happened”

Example:

Job 1: Setup
✗ FAILED

Job 2: Unit Tests
○ UNREACHED

Job 3: Integration Tests
○ UNREACHED

Legend:

✓ passed
✗ failed
○ unreached

Beside this, show a tiny terminal-style CI log:

CI LOG

[Job: Setup]
[Step: Install dependencies]

$ pip install -r requirements.txt

ERROR:
required httpx version missing /
dependency mismatch

exit code 1

✗ FAILED

The purpose is to teach the reader that:

WORKFLOW = what CI is configured to execute

EXECUTION / LOG = what actually happened when the workflow ran

A failed earlier job can prevent downstream validations from executing.

------------------------------------------------------------
HISTORICAL REPAIR TRAJECTORY
------------------------------------------------------------

Below or beside the CI example, show the completed historical repair:

Failing Revision R−
       ↓
      C1
       ↓
      C2
       ↓
      C3
       ↓
Passing Revision R+

Use this concrete example:

C1 requirements.txt
+ httpx==0.25.0

C2 src/client.py
- s = httpx.Session()
+ s = httpx.Client()

C3 utils.py
rename helper

Mark C3 as potentially unrelated.

The final passing revision contains the net result of the development process.

Show that the completed PR contains at least two CI problems:

P1: Missing dependency / dependency problem

P2: API incompatibility

The running example should suggest a relation such as:

P1
Missing dependency
     ↓ enables / exposes
P2
API incompatibility

The exact relation should be presented as historical CI evidence, not as a generic rule.

Add a central question:

“How can reusable problem-level repair experience be recovered from this entangled PR history?”


################################################################
STAGE 1: MULTI-VIEW RECONSTRUCTION
################################################################

Use the exact major heading:

1. MULTI-VIEW RECONSTRUCTION

The goal of this stage is to transform the completed PR-level repair history into supported problem-level repair records.

============================================================
1A. ORGANIZE CI EVIDENCE
============================================================

Show:

Raw CI Logs
+
CI Workflow
        ↓
ORGANIZE CI EVIDENCE

Represent the result as:

O = (F, V, G)

where visually:

F = Observed Failures
V = Validation Status / Context
G = Workflow Dependencies

Example:

F:
P1 dependency failure

V:
Setup = FAILED
Unit Tests = UNREACHED
Integration = UNREACHED

G:

Setup
 ↓
Unit Tests
 ↓
Integration Tests

Show that organized CI evidence includes:

- diagnostic messages
- originating commands
- workflow locations
- referenced artifacts
- validation status
- job/step dependencies

IMPORTANT:

Place ORGANIZED CI EVIDENCE visually so that it clearly supports BOTH the Development View and Endpoint View.

It is the shared evidence anchor for the two views.


============================================================
1B. TWO COMPLEMENTARY VIEWS
============================================================

Do NOT create a separate box or stage called:

“Bidirectional Change Attribution”

Instead, directly show the TWO complementary views under:

MULTI-VIEW RECONSTRUCTION

Use:

DEVELOPMENT VIEW                         ENDPOINT VIEW
     forward →                              ← backward

The opposite arrow directions should visually communicate the two directions.


------------------------------------------------------------
DEVELOPMENT VIEW
------------------------------------------------------------

Title:

DEVELOPMENT VIEW

Small directional annotation:

forward →

Question:

“How did candidate repairs evolve?”

Show the chronological developer history:

C1 requirements.txt                       P1
+ httpx==0.25.0

        ↓

C2 src/client.py                          P2
- s = httpx.Session()
+ s = httpx.Client()

        ↓

C3 utils.py
rename helper                         unrelated

Make this a visible commit sequence:

C1 → C2 → C3

Show that the Development View traces how potential repairs were:

- introduced
- revised
- removed
- retained

If useful, show one small faded/crossed-out intermediate change to demonstrate an abandoned or revised repair attempt.

The Development View should answer:

“How did potential repairs develop through the commit history, and which observed CI problems are they supported as addressing?”

Changes without a supported connection to a CI problem should be excluded from the repair association.

However, revised or removed changes may remain visible as development history.


------------------------------------------------------------
ENDPOINT VIEW
------------------------------------------------------------

Title:

ENDPOINT VIEW

Small directional annotation:

← backward

Question:

“Which final changes are supported as repairs?”

Show:

Failing Revision R−
        +
Passing Revision R+
        ↓
Final Endpoint Patch

Δ = diff(R−, R+)

Example final net changes:

+ httpx==0.25.0                         P1

- s = httpx.Session()
+ s = httpx.Client()                    P2

rename helper                           unsupported /
                                        unrelated

Show that the endpoint view:

1. starts from changes retained in the passing revision,
2. groups related changes where appropriate,
3. compares them against the organized CI evidence,
4. reasons backward to the CI problems they may address,
5. excludes changes with no supported connection to a CI problem.

Use the question:

“Which retained changes are supported by CI evidence as addressing each problem?”

IMPORTANT:

The Endpoint View is NOT simply “the final solution.”

It is a reconstruction view based on the cumulative patch between R− and R+.


============================================================
1C. EVIDENCE RECONCILIATION
============================================================

Merge the Development View and Endpoint View visually:

DEVELOPMENT VIEW             ENDPOINT VIEW
       \                           /
        \                         /
         \                       /
          ↓                     ↓
          EVIDENCE RECONCILIATION

Use the exact heading:

EVIDENCE RECONCILIATION

Inside the box ask:

“Which problem-repair associations are supported?”

Do NOT use:

“Which change fixed which problem?”

because the methodology reconstructs supported problem-repair associations that may span multiple files or commits.

Show that records from the two views can be:

Equivalent
Complementary
Conflicting
Distinct

Then show reconciliation against historical artifacts:

Organized CI Evidence
Commit History
Endpoint Patch
Passing Repository State

Unsupported fields or associations are removed.

A record appearing in only one view may still be retained when supported by the available historical evidence.

Use the running example:

Development View:
C1 → P1
C2 → P2
C3 → unrelated

Endpoint View:
C1-equivalent retained change → P1
C2-equivalent retained change → P2
C3 has no supported CI connection

Reconciled:

C1 ↔ P1     ✓ supported
C2 ↔ P2     ✓ supported
C3          ✗ discard from repair record


============================================================
1D. PROBLEM-LEVEL REPAIR RECORDS
============================================================

The direct output of reconciliation should be:

PROBLEM-LEVEL REPAIR RECORDS

Make this transformation visually prominent:

ENTANGLED PR-LEVEL HISTORY
          ↓
MULTI-VIEW RECONSTRUCTION
          ↓
PROBLEM-LEVEL REPAIR RECORDS

Show two compact example records.

RECORD 1

Problem:
P1 Missing dependency

Inferred cause:
required dependency/version unavailable

Repair:
add/update required dependency

Affected artifact:
requirements.txt

Validation:
dependency installation / relevant CI validation

Evidence:
CI evidence + C1 + endpoint patch


RECORD 2

Problem:
P2 API incompatibility

Inferred cause:
code uses incompatible API

Repair:
replace Session with compatible Client API

Affected artifact:
src/client.py

Validation:
relevant tests

Evidence:
CI evidence + C2 + endpoint patch

Show a historical problem relation if supported:

P1 Missing dependency
       ↓ enables / prerequisite
P2 API incompatibility

IMPORTANT:

Each record represents ONE failure mechanism and its associated repair.

A repair may span multiple files or commits.

One commit may also contribute to multiple records.

Do not visually imply a strict one-commit-to-one-problem assumption.


################################################################
STAGE 2: REPAIR EXPERIENCE AT THREE LEVELS
################################################################

Use the exact heading:

2. REPAIR EXPERIENCE AT THREE LEVELS

Take each reconstructed problem-level repair record and represent it at three linked abstraction levels.

Show:

             SAME REPAIR
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
       L1       L2       L3

or a progressive abstraction:

L1 → L2 → L3

Make clear that all three representations refer to the SAME underlying historical repair and remain linked to supporting evidence.


============================================================
L1: CONCRETE REPAIR EXPERIENCE
============================================================

Label:

L1
CONCRETE REPAIR EXPERIENCE

Preserve:

- observed failure
- inferred cause
- affected files/symbols
- actual developer repair
- validation evidence
- commit evidence

Example:

Problem:
API incompatibility

File:
src/client.py

Change:
httpx.Session()
→
httpx.Client()

Validation:
relevant CI validation

Evidence:
C2 + endpoint patch


============================================================
L2: CONTEXTUAL REPAIR STRATEGY
============================================================

Label:

L2
CONTEXTUAL REPAIR STRATEGY

Abstract away literal file paths and symbol names where possible while preserving repository-relevant roles and conditions.

Example:

“When a dependency/API upgrade makes a client-construction API incompatible, identify the affected consumer, migrate it to the supported client interface, and rerun the affected validation.”

Show:

- failure mechanism
- diagnostic signals
- artifact relationships
- repair steps
- validation procedure
- applicability-relevant context


============================================================
L3: TRANSFERABLE REPAIR PATTERN
============================================================

Label:

L3
TRANSFERABLE REPAIR PATTERN

Remove repository-specific identifiers and implementation details.

Example:

“API incompatibility after dependency change:
identify obsolete API usage,
replace it with the supported interface,
and validate downstream consumers.”

Show:

- general failure mechanism
- diagnostic signals
- artifact roles
- repair principle
- applicability conditions
- historically observed follow-on problems


============================================================
EXPERIENCE COLLECTION
============================================================

Show the resulting historical experience organized by level:

L1 Experience
L2 Experience
L3 Experience

Do not imply that L1, L2, and L3 are independent repairs.

They are three linked representations of the same reconstructed repair experience.


################################################################
STAGE 3: DYNAMIC EXPERIENCE REUSE
################################################################

Use the exact heading:

3. DYNAMIC EXPERIENCE REUSE

Clearly transition from historical learning to a NEW target CI failure.

============================================================
3A. NEW TARGET CI FAILURE
============================================================

Show:

NEW TARGET REPAIR

Target Failing Revision Rx

        ↓

Target CI Workflow + CI Log

Example:

Job 1: Setup
✓ PASS

Job 2: Unit Tests
✗ FAIL

Error:
client initialization / API incompatibility

Job 3: Integration Tests
○ UNREACHED

IMPORTANT:

Only show information available at the failing target revision.

Do NOT show the target developer patch or future passing state.


============================================================
3B. TARGET-CONDITIONED PROBLEM SET
============================================================

Show:

Target CI Evidence
        ↓
ORGANIZE
        ↓
DECOMPOSE INTO ATOMIC PROBLEMS

Example observed problem:

T1:
API incompatibility

Failure signals:
client initialization failure

Affected artifact:
client-related code

Validation:
unit tests


============================================================
3C. LEVEL-SPECIFIC EXPERIENCE RETRIEVAL
============================================================

For EACH observed target problem, show three separate retrieval queries:

                     T1
              /       |       \
             ↓        ↓        ↓
          L1 Query  L2 Query  L3 Query
             ↓        ↓        ↓
             L1       L2       L3
          Experience Experience Experience

Make it visually explicit that retrieval occurs:

- separately for each target problem
- separately at each abstraction level

Do NOT depict retrieval of an entire historical PR as a single example.


============================================================
3D. APPLICABILITY FILTERING
============================================================

Show:

Retrieved Candidate Experience
          ↓
APPLICABILITY FILTERING

Compare each candidate against:

- target CI evidence
- target repository state
- failure mechanism
- repair conditions
- assumptions

Show:

Candidate A
✓ applicable

Candidate B
✗ semantically similar but incompatible

Candidate C
✓ applicable

Only applicable historical experience becomes guidance for the repair agent.


============================================================
3E. DEPENDENCY-DERIVED PROBLEMS
============================================================

Show that historical experience may contain prerequisite relations.

Example:

Observed Target Problem T1
        ↓
matches historical problem
        ↓
Historical relation:
P1 → P2
        ↓
Potential downstream problem
        ↓
T2

Use a small label:

“Earlier failures may hide downstream problems.”

This should explain why an UNREACHED CI validation may still correspond to a problem that becomes visible after an earlier problem is repaired.


============================================================
3F. RECURRING-PROBLEM SET
============================================================

Show a second compact source:

RECURRING PROBLEMS

Derived from repeated similar reconstructed problems in the same repository and workflow.

Then show:

Target-Conditioned Problems
          +
Recurring Problems
          ↓
MERGE + ORDER


============================================================
3G. CONSOLIDATION AND ORDERING
============================================================

Show:

Observed Problems
+
Dependency-Derived Problems
+
Recurring Problems
        ↓
MERGE + ORDER
        ↓
FINAL PROBLEM SEQUENCE

Example:

<T1, T2>

Ordering should reflect:

- target workflow dependencies
- historical prerequisite relations
- CI execution order when otherwise unconstrained


============================================================
3H. SEQUENTIAL REPAIR AND VALIDATION
============================================================

Show a repair agent working in a shared working tree.

Label:

REPAIR AGENT

Visual flow:

Failing Target State R(1)
        ↓
Repair T1

Inputs:
T1 problem-specific CI evidence
+
T1 historical repair guidance

        ↓
Repository State R(2)

        ↓
Repair T2

Inputs:
T2 problem-specific CI evidence
+
T2 historical repair guidance

        ↓
Repository State R(3)

Make clear:

“Previous repairs remain in the shared working tree.”

Add a small note:

“Problem-specific validation when feasible”

Do NOT show the complete CI workflow being rerun after every individual problem.


============================================================
3I. FINAL FULL CI VALIDATION
============================================================

At the far right show:

FINAL FULL CI WORKFLOW

✓ Setup
✓ Unit Tests
✓ Integration Tests

        ↓

CI-PASSING REPOSITORY

Use a strong final visual:

✓ ALL CI JOBS AND VALIDATIONS PASS


################################################################
REQUIRED VISUAL HIERARCHY
################################################################

The most visually important portion of Stage 1 should be:

                 ORGANIZED CI EVIDENCE
                         │
               shared evidence anchor
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
      DEVELOPMENT VIEW         ENDPOINT VIEW
         forward →              ← backward

      C1 → C2 → C3            Δ = diff(R−,R+)

              │                     │
              └──────────┬──────────┘
                         ▼
              EVIDENCE RECONCILIATION
                         ▼
              PROBLEM-LEVEL REPAIR
                     RECORDS

Do not add a separate methodology component called
“Bidirectional Change Attribution.”

The forward Development View and backward Endpoint View should visually communicate the complementary directions.


################################################################
COMPLETE END-TO-END VISUAL FLOW
################################################################

The complete figure should approximately read:

HISTORICAL CI REPAIR

Developer Changes
      ↓
Pull Request
      ↓
CI Workflow
      ↓
Jobs → Steps → Commands
      ↓
CI Execution + CI Log
      ↓
Failing Revision R−
      ↓
C1 → C2 → C3
      ↓
Passing Revision R+


1. MULTI-VIEW RECONSTRUCTION

CI Logs + Workflow
      ↓
Organized CI Evidence
      ↓
┌─────────────────────────────────┐
│                                 │
▼                                 ▼
Development View             Endpoint View
forward →                    ← backward
C1 → C2 → C3                 endpoint patch Δ
│                                 │
└──────────────┬──────────────────┘
               ↓
      Evidence Reconciliation
               ↓
      Problem-Level Repair Records


2. REPAIR EXPERIENCE AT THREE LEVELS

Problem-Level Repair Record
               ↓
       L1 → L2 → L3
       Concrete
          → Contextual
             → Transferable


3. DYNAMIC EXPERIENCE REUSE

New Target CI Failure
               ↓
       Organize CI Evidence
               ↓
      Atomic Target Problems
               ↓
    Per-Problem L1/L2/L3 Retrieval
               ↓
       Applicability Filtering
               ↓
   Dependency-Derived Problems
               +
       Recurring Problems
               ↓
          Merge + Order
               ↓
       Sequential Repair
               ↓
        Final Full CI Run
               ↓
               ✓
        CI-PASSING REPOSITORY


################################################################
KEY MESSAGE
################################################################

The figure should make the following story understandable without requiring the reader to read the full methodology first:

A completed CI repair contains valuable human repair experience, but the experience is entangled across multiple CI problems, commits, intermediate attempts, and unrelated development changes.

ReCAP first organizes CI execution evidence.

It then reconstructs problem-level repair experience through two complementary views:

- the Development View traces forward through the commit history to understand how candidate repairs were introduced, revised, removed, or retained;

- the Endpoint View reasons backward from the changes retained in the final passing revision to determine which CI problems those changes are supported as addressing.

The candidate problem-repair associations from the two views are reconciled against the historical evidence.

The resulting problem-level repair records are represented at three abstraction levels:

L1 Concrete Repair Experience
L2 Contextual Repair Strategy
L3 Transferable Repair Pattern

For a new target CI failure, ReCAP decomposes the observed CI evidence into problems, retrieves historical experience separately for each problem and abstraction level, filters it for applicability, incorporates dependency-derived and recurring problems, orders the resulting problem set, and provides problem-specific historical guidance to a repair agent.

The agent repairs the problems sequentially in a shared working tree.

After all problems have been processed, the complete CI workflow is rerun.

The repair succeeds only when all CI jobs and validations pass.


################################################################
DESIGN REQUIREMENTS
################################################################

Use a publication-quality visual style suitable for ICLR/NeurIPS/ICML.

- wide horizontal layout
- clean white background
- vector-style graphics
- flat 2D design
- restrained academic color palette
- compact text
- consistent typography
- strong visual hierarchy
- readable at paper width
- terminal-style CI log boxes
- small code-diff snippets
- small commit nodes
- clear directional arrows
- visually distinguish historical reconstruction from target-time reuse

Use consistent visual identities for:

CI evidence
developer commits
endpoint patch
problem labels
repair records
L1/L2/L3 experience
target problems
repair agent
validation results

Use color plus shape/labels, not color alone.

The figure must remain understandable in grayscale.

Avoid:

- photorealism
- 3D graphics
- decorative AI-brain icons
- excessive GitHub branding
- large blocks of code
- unnecessary prose
- excessive arrows
- generic architecture boxes without the running example


################################################################
FINAL SELF-CHECK
################################################################

Before finalizing the figure, verify that an AI/ML reader can understand:

1. What a CI workflow is.
2. The hierarchy Workflow → Job → Step → Command.
3. The difference between a CI workflow and CI execution log.
4. What passed, failed, and unreached validations mean.
5. Why a completed PR is an entangled source of repair experience.
6. What organized CI evidence contains.
7. What the Development View analyzes.
8. Why the Development View is forward.
9. What the Endpoint View analyzes.
10. Why the Endpoint View is backward.
11. How both views use organized CI evidence.
12. What Evidence Reconciliation does.
13. How an entangled PR becomes problem-level repair records.
14. What L1, L2, and L3 represent.
15. How experience is retrieved separately for each target problem.
16. Why applicability filtering is needed.
17. How historical dependencies can reveal potentially hidden downstream problems.
18. How recurring problems contribute to the target problem set.
19. How problems are merged and ordered.
20. How the repair agent processes them sequentially.
21. Why the complete CI workflow is executed at the end.
22. That success means all CI jobs and validations pass.