# Agent Instructions

Dear Agent, this is a project in which we should optimize for correctness, clean implementation, passing tests and clear explanation and documentation. 

# Hard Rules
- Do not run `litmus init` or `litmus submit` unless I explicitly tell you to do so. Never. 
- Do not delete, weaken, bypass or fake tests. 
- Do not hardcode any visible exaples or hidden-caseguessses. 
- Do not rewrite unrelated parts of the project. No fluff. 
- Do not add dpeendencies unless abslutely necessary. 
- Do not leave debug logs, temporary files and documentation, secreets or unusuaed or obsolete code. 
- Commit frequently to github with small, meaningful commits with brief commit messages but complete with good coding practices. 
- Do not create unnecessary .md files. 
- Absolutely no emdashes in code. 

# Workflow
1. Read the prompt fully. 
2. Inspect the project structure, README, package files, source code, and tests. 
3. Make a short plan before editing. 
4. Implement the smallest correct change but still writing the most efficient code possible. 
5. Run the reltevasts fter eachnt menaingful change. 
6. Review the diff before finalizing. 
7. Commmit each meanigful step. 

## Programming Principles

Write code that is:

- Correct before clever
- Simple before abstract
- Readable before compressed
- Explicit before magical
- Tested before trusted
- Maintainable before over-engineered
- Prefer boring, reliable code over impressive-looking bad code. 

Use these principles:


- KISS: keep the solution simple.
- YAGNI: do not build features that were not requested.
- DRY: avoid meaningful duplication, but do not create bad abstractions too early.
- Single responsibility: each function/module should do one clear thing.
- Fail clearly: handle errors intentionally and make failures understandable.
- Minimal surface area: expose only what is needed.
- Type safety: use existing typing conventions where applicable.
- Determinism: avoid flaky behavior, race conditions, and hidden global state.
- Security: never leak secrets, never trust user input blindly, and avoid unsafe eval-style logic.

# Git RUles
Commmit and push frequently. Good commit exaples:

```sh
git add .
git commit -m "Inspect project structure and assessment requirements and draft starter files."
git push origin main.
```