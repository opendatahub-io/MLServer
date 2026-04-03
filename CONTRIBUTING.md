# Contributing to MLServer

Opening a PR
------------
* Fork the repository from `SeldonIO` into local Github account. 
* create a branch from the master of the forked repository.<br> `git checkout -b <branch>`.
<br> branches can be named as `bug|feat|doc|`_`<desc>`_`<optional params>`
* make changes, and raise a PR from local repository `<branch>` to main repository `master`.
* make sure that your branch is always uptodate (you may use rebase to resolve conflicts) with **SeldonIO/MLServer** `master` branch. <br>
```bash
   git remote add upstream https://github.com/SeldonIO/MLServer.git 
   git fetch upstream
   git checkout <your_branch_name>
   git rebase upstream/master
```

Installation for Contributing
------------
- git clone the forked repository: <br>`git clone <your-repo>/MLServer <your-folder>` 
- setup `mlserver`: ```pip install .```
- run examples in debugging mode and verify execution taking one to breakpoints in one's development branch

Running Tests
------------

**Recommended approach:** Use `make test` or `tox` to run the full test suite. These commands handle test isolation and execution order correctly.

```bash
# Run all tests (recommended)
make test

# Or using tox directly
tox -e mlserver
```

### Running Tests Manually

If running tests manually with pytest, be aware of these constraints:

**Most tests - use parallel execution:**
```bash
# Run main test suite (excludes special test directories)
pytest tests/ -n auto \
  --ignore=tests/metrics \
  --ignore=tests/kafka \
  --ignore=tests/parallel \
  --ignore=tests/grpc \
  --ignore=tests/env \
  --ignore=tests/cli
```

**Special test suites - run separately (sequential):**
```bash
# ✅ Correct - run individually without -n auto
pytest tests/metrics/
pytest tests/kafka
pytest tests/parallel/
pytest tests/grpc/
pytest tests/env/
pytest tests/cli/

# ❌ Incorrect - mixing with other tests or using -n auto
pytest tests/  # Mixes special tests with main suite
pytest tests/parallel/ -n auto  # Race conditions
```

**Why these run separately:**
- `tests/parallel/` - Spawns worker processes, modifies environment variables (race conditions with `-n auto`)
- `tests/kafka/` - Requires Docker daemon running
- `tests/metrics/`, `tests/grpc/`, `tests/env/`, `tests/cli/` - Flaky when run in parallel with main test suite

See `tox.ini` for how test isolation is handled in the official test suite.

Raising a PR
------------
- Choose a default PR template/templates available underneath `/docs/PULL_REQUEST_TEMPLATE/` as a `template` query param. 

_Before opening a pull request_ consider:

- Is the change important and ready enough to ask the community to spend time reviewing?
- Have you searched for existing, related issues and pull requests?
- Is the change being proposed clearly explained and motivated?

When you contribute code, you affirm that the contribution is your original work and that you
license the work to the project under the project's open source license. Whether or not you
state this explicitly, by submitting any copyrighted material via pull request, email, or
other means you agree to license the material under the project's open source license and
warrant that you have the legal authority to do so.
