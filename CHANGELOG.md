# Changelog


<a id="v1.7.1+rhaiv.9"></a>
## [AIPCC release for MLServer v1.7.1+rhaiv.9](https://github.com/opendatahub-io/MLServer/releases/tag/v1.7.1+rhaiv.9) - 2026-05-08

<!-- Release notes generated using configuration in .github/release.yml at rhoai-staging -->

## What's Changed
* Update OWNERS File by [@brettmthompson](https://github.com/brettmthompson) in [#3](https://github.com/opendatahub-io/MLServer/pull/3)
* Refactor Dockerfile to remove non-required runtimes from MLServer image by [@Snomaan6846](https://github.com/Snomaan6846) in [#1](https://github.com/opendatahub-io/MLServer/pull/1)
* Adding Workflow To Merge Release Branches to rhoai-staging Branch by [@brettmthompson](https://github.com/brettmthompson) in [#2](https://github.com/opendatahub-io/MLServer/pull/2)
* Stabilize CI Tests by [@brettmthompson](https://github.com/brettmthompson) in [#5](https://github.com/opendatahub-io/MLServer/pull/5)
* updating owners aliases file by [@brettmthompson](https://github.com/brettmthompson) in [#10](https://github.com/opendatahub-io/MLServer/pull/10)
* Remove conda reference from Dockerfile by [@Snomaan6846](https://github.com/Snomaan6846) in [#6](https://github.com/opendatahub-io/MLServer/pull/6)
* remove unnecessary files and add ignore files for sync by [@Jooho](https://github.com/Jooho) in [#17](https://github.com/opendatahub-io/MLServer/pull/17)
* Adding Required Workflows by [@brettmthompson](https://github.com/brettmthompson) in [#12](https://github.com/opendatahub-io/MLServer/pull/12)
* Add build arguments to parameterize Dockerfile base images by [@Snomaan6846](https://github.com/Snomaan6846) in [#13](https://github.com/opendatahub-io/MLServer/pull/13)
* Sync master to release-1.7.x by [@brettmthompson](https://github.com/brettmthompson) in [#18](https://github.com/opendatahub-io/MLServer/pull/18)
* preserving commit history in the release to staging sync workflow by [@brettmthompson](https://github.com/brettmthompson) in [#19](https://github.com/opendatahub-io/MLServer/pull/19)
* Revert "Sync master to release-1.7.x ([#18](https://github.com/opendatahub-io/MLServer/issues/18))" by [@brettmthompson](https://github.com/brettmthompson) in [#21](https://github.com/opendatahub-io/MLServer/pull/21)
* Sync master to release-1.7.x by [@brettmthompson](https://github.com/brettmthompson) in [#22](https://github.com/opendatahub-io/MLServer/pull/22)
* Fix merging conflict for syncing from release-1.7.x to  rhoai-staging by [@Jooho](https://github.com/Jooho) in [#23](https://github.com/opendatahub-io/MLServer/pull/23)
* Cleanup rhoai-staging by [@brettmthompson](https://github.com/brettmthompson) in [#24](https://github.com/opendatahub-io/MLServer/pull/24)
* improvements to sync workflow by [@brettmthompson](https://github.com/brettmthompson) in [#25](https://github.com/opendatahub-io/MLServer/pull/25)
* Cherry-pick from master to release branch by [@Snomaan6846](https://github.com/Snomaan6846) in [#30](https://github.com/opendatahub-io/MLServer/pull/30)
* Cherry-pick from master to rhoai-staging branch by [@Snomaan6846](https://github.com/Snomaan6846) in [#31](https://github.com/opendatahub-io/MLServer/pull/31)
* Downgrade odh release version to 3.2 by [@Snomaan6846](https://github.com/Snomaan6846) in [#34](https://github.com/opendatahub-io/MLServer/pull/34)
* Harden Event Loop Logic by [@brettmthompson](https://github.com/brettmthompson) in [#35](https://github.com/opendatahub-io/MLServer/pull/35)
* converting all occurences of get_event_loop to get_running_loop ([#35](https://github.com/opendatahub-io/MLServer/issues/35)) by [@brettmthompson](https://github.com/brettmthompson) in [#36](https://github.com/opendatahub-io/MLServer/pull/36)
* Update Tekton files to version odh-v3.2 by [@odh-devops-app](https://github.com/odh-devops-app)[bot] in [#37](https://github.com/opendatahub-io/MLServer/pull/37)
* cherry-pick-event-loop-changes by [@brettmthompson](https://github.com/brettmthompson) in [#38](https://github.com/opendatahub-io/MLServer/pull/38)
* chore(konflux): Bump release tag to odh-v3.3 by [@github-actions](https://github.com/github-actions)[bot] in [#39](https://github.com/opendatahub-io/MLServer/pull/39)
* Update MLServer and Runtimes version to 1.7.1+rhai4 by [@Snomaan6846](https://github.com/Snomaan6846) in [#41](https://github.com/opendatahub-io/MLServer/pull/41)
* Upgrade sklearn images used in testing to the latest version by [@brettmthompson](https://github.com/brettmthompson) in [#44](https://github.com/opendatahub-io/MLServer/pull/44)
* Disable Security Scan In ODH by [@brettmthompson](https://github.com/brettmthompson) in [#45](https://github.com/opendatahub-io/MLServer/pull/45)
* Podman Support for MLServer Tests by [@brettmthompson](https://github.com/brettmthompson) in [#43](https://github.com/opendatahub-io/MLServer/pull/43)
* Cherry pick chores to release by [@brettmthompson](https://github.com/brettmthompson) in [#46](https://github.com/opendatahub-io/MLServer/pull/46)
* Cherry pick chores to staging by [@brettmthompson](https://github.com/brettmthompson) in [#47](https://github.com/opendatahub-io/MLServer/pull/47)
* Update MLServer and Runtimes version to 1.7.1+rhai5 by [@Snomaan6846](https://github.com/Snomaan6846) in [#48](https://github.com/opendatahub-io/MLServer/pull/48)
* Add Dockerfile.konflux for rhoai releases by [@Snomaan6846](https://github.com/Snomaan6846) in [#50](https://github.com/opendatahub-io/MLServer/pull/50)
* add pipelineruns for odh ci builds by [@MohammadiIram](https://github.com/MohammadiIram) in [#66](https://github.com/opendatahub-io/MLServer/pull/66)
* Making poetry version configurable in tests workflow by [@brettmthompson](https://github.com/brettmthompson) in [#72](https://github.com/opendatahub-io/MLServer/pull/72)
* fix: XGBoost model loading issue with modelcar by [@Snomaan6846](https://github.com/Snomaan6846) in [#70](https://github.com/opendatahub-io/MLServer/pull/70)
* fix: XGBoost model loading issue with modelcar ([#70](https://github.com/opendatahub-io/MLServer/issues/70)) by [@Snomaan6846](https://github.com/Snomaan6846) in [#78](https://github.com/opendatahub-io/MLServer/pull/78)
* Cherry pick rhoaieng 46109 by [@Snomaan6846](https://github.com/Snomaan6846) in [#79](https://github.com/opendatahub-io/MLServer/pull/79)
* Make permissions explicit in merge workflow by [@brettmthompson](https://github.com/brettmthompson) in [#76](https://github.com/opendatahub-io/MLServer/pull/76)
* feat(runtimes): add ONNX runtime support (mlserver_onnx) by [@Snomaan6846](https://github.com/Snomaan6846) in [#73](https://github.com/opendatahub-io/MLServer/pull/73)
* feat(runtimes): add ONNX runtime support (mlserver_onnx) ([#73](https://github.com/opendatahub-io/MLServer/issues/73)) by [@Snomaan6846](https://github.com/Snomaan6846) in [#84](https://github.com/opendatahub-io/MLServer/pull/84)
* feat(runtimes): add ONNX runtime support (mlserver_onnx) ([#73](https://github.com/opendatahub-io/MLServer/issues/73)) by [@Snomaan6846](https://github.com/Snomaan6846) in [#85](https://github.com/opendatahub-io/MLServer/pull/85)
* Update MLServer and Runtimes version to 1.7.1+rhai6 by [@Snomaan6846](https://github.com/Snomaan6846) in [#86](https://github.com/opendatahub-io/MLServer/pull/86)
* CI: Add AIPCC wheels requirements file generation script and github workflow by [@Snomaan6846](https://github.com/Snomaan6846) in [#88](https://github.com/opendatahub-io/MLServer/pull/88)
* ci: Add AIPCC wheels requirements file generation script and github w… by [@Snomaan6846](https://github.com/Snomaan6846) in [#91](https://github.com/opendatahub-io/MLServer/pull/91)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#92](https://github.com/opendatahub-io/MLServer/pull/92)
* onnx runtime: align version constraints and stabilize test model metadata by [@Snomaan6846](https://github.com/Snomaan6846) in [#95](https://github.com/opendatahub-io/MLServer/pull/95)
* onnx runtime: align version constraints and stabilize test model metadata by [@Snomaan6846](https://github.com/Snomaan6846) in [#99](https://github.com/opendatahub-io/MLServer/pull/99)
* onnx runtime: align version constraints and stabilize test model metadata by [@Snomaan6846](https://github.com/Snomaan6846) in [#98](https://github.com/opendatahub-io/MLServer/pull/98)
* updating push job to create odh-v3.4-EA1 tag by [@brettmthompson](https://github.com/brettmthompson) in [#101](https://github.com/opendatahub-io/MLServer/pull/101)
* updating push job to create odh-v3.4-EA2 tag by [@brettmthompson](https://github.com/brettmthompson) in [#102](https://github.com/opendatahub-io/MLServer/pull/102)
* bumping tag in tekton push job to odh-v3.4 by [@brettmthompson](https://github.com/brettmthompson) in [#103](https://github.com/opendatahub-io/MLServer/pull/103)
* Add mlserver-onnx to requirements-config.json by [@Snomaan6846](https://github.com/Snomaan6846) in [#107](https://github.com/opendatahub-io/MLServer/pull/107)
* Add support for onnx model format for in Dockerfile by [@Snomaan6846](https://github.com/Snomaan6846) in [#106](https://github.com/opendatahub-io/MLServer/pull/106)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#108](https://github.com/opendatahub-io/MLServer/pull/108)
* update dockerfile.konflux to include labels ([#66](https://github.com/opendatahub-io/MLServer/issues/66)) by [@Snomaan6846](https://github.com/Snomaan6846) in [#109](https://github.com/opendatahub-io/MLServer/pull/109)
* Various improvements to how MLServer gets built by [@RH-steve-grubb](https://github.com/RH-steve-grubb) in [#94](https://github.com/opendatahub-io/MLServer/pull/94)
* chore(renovate): add renovate config for aipcc base image updates on rhoai-staging by [@Snomaan6846](https://github.com/Snomaan6846) in [#111](https://github.com/opendatahub-io/MLServer/pull/111)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#112](https://github.com/opendatahub-io/MLServer/pull/112)
* chore: Use single reference of base image by [@Snomaan6846](https://github.com/Snomaan6846) in [#113](https://github.com/opendatahub-io/MLServer/pull/113)
* chore(renovate): Update renovate config to remove includePaths config by [@Snomaan6846](https://github.com/Snomaan6846) in [#114](https://github.com/opendatahub-io/MLServer/pull/114)
* chore(renovate): add daily schedule for dockerfile manager for renovate by [@Snomaan6846](https://github.com/Snomaan6846) in [#115](https://github.com/opendatahub-io/MLServer/pull/115)
* fix: ONNX model loading failure with KServe Modelcar symlinks by [@Jooho](https://github.com/Jooho) in [#117](https://github.com/opendatahub-io/MLServer/pull/117)
* [release-1.7.x] fix: ONNX model loading failure with KServe Modelcar symlinks by [@openshift-cherrypick-robot](https://github.com/openshift-cherrypick-robot) in [#118](https://github.com/opendatahub-io/MLServer/pull/118)
* [rhoai-staging] fix: ONNX model loading failure with KServe Modelcar symlinks by [@openshift-cherrypick-robot](https://github.com/openshift-cherrypick-robot) in [#119](https://github.com/opendatahub-io/MLServer/pull/119)
* Align Dockerfile.konflux improvements and  update aipcc base image for 3.4.0 release by [@Snomaan6846](https://github.com/Snomaan6846) in [#120](https://github.com/opendatahub-io/MLServer/pull/120)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#121](https://github.com/opendatahub-io/MLServer/pull/121)
* Cherry pick from master to release 1.7.x by [@Snomaan6846](https://github.com/Snomaan6846) in [#122](https://github.com/opendatahub-io/MLServer/pull/122)
* chore(ci): harden release-to-staging sync workflow merge and policy handling by [@Snomaan6846](https://github.com/Snomaan6846) in [#123](https://github.com/opendatahub-io/MLServer/pull/123)
* [release-1.7.x] chore(ci): harden release-to-staging sync workflow merge and policy handling by [@openshift-cherrypick-robot](https://github.com/openshift-cherrypick-robot) in [#124](https://github.com/opendatahub-io/MLServer/pull/124)
* Sync release-1.7.x to rhoai-staging by [@github-actions](https://github.com/github-actions)[bot] in [#125](https://github.com/opendatahub-io/MLServer/pull/125)
* chore(ci): require manual pyproject sync policy acknowledgement in release sync workflow by [@Snomaan6846](https://github.com/Snomaan6846) in [#126](https://github.com/opendatahub-io/MLServer/pull/126)
* [release-1.7.x] chore(ci): require manual pyproject sync policy acknowledgement in release sync workflow by [@openshift-cherrypick-robot](https://github.com/openshift-cherrypick-robot) in [#127](https://github.com/opendatahub-io/MLServer/pull/127)
* Make runtime allowlist flexible by [@brettmthompson](https://github.com/brettmthompson) in [#110](https://github.com/opendatahub-io/MLServer/pull/110)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#128](https://github.com/opendatahub-io/MLServer/pull/128)
* chore: install mlserver wheel prior to runtime wheels by [@Snomaan6846](https://github.com/Snomaan6846) in [#131](https://github.com/opendatahub-io/MLServer/pull/131)
* Sync master to release by [@Snomaan6846](https://github.com/Snomaan6846) in [#130](https://github.com/opendatahub-io/MLServer/pull/130)
* Sync release-1.7.x to rhoai-staging by [@github-actions](https://github.com/github-actions)[bot] in [#132](https://github.com/opendatahub-io/MLServer/pull/132)
* master to release sync and lock the poetry.lock files of MLServer and Runtimes to 1.7.1 version by [@Snomaan6846](https://github.com/Snomaan6846) in [#135](https://github.com/opendatahub-io/MLServer/pull/135)
* Sync release-1.7.x to rhoai-staging by [@github-actions](https://github.com/github-actions)[bot] in [#136](https://github.com/opendatahub-io/MLServer/pull/136)
* chore: align konflux Dockerfile and bump MLServer/runtimes to 1.7.1+rhaiv.8 by [@Snomaan6846](https://github.com/Snomaan6846) in [#133](https://github.com/opendatahub-io/MLServer/pull/133)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#138](https://github.com/opendatahub-io/MLServer/pull/138)
* [release-1.7.x] chore(ci): add tide/merge-method-merge label for PRs generated by release to rhoai-staging sync workflow by [@openshift-cherrypick-robot](https://github.com/openshift-cherrypick-robot) in [#140](https://github.com/opendatahub-io/MLServer/pull/140)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#141](https://github.com/opendatahub-io/MLServer/pull/141)
* [release-1.7.x] fix: prevent redundant InferencePool spawning for same inference_pool by [@openshift-cherrypick-robot](https://github.com/openshift-cherrypick-robot) in [#145](https://github.com/opendatahub-io/MLServer/pull/145)
* Sync release-1.7.x to rhoai-staging by [@github-actions](https://github.com/github-actions)[bot] in [#146](https://github.com/opendatahub-io/MLServer/pull/146)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#151](https://github.com/opendatahub-io/MLServer/pull/151)
* [release-1.7.x] chore: drop Python 3.9 support and modernize to 3.10+ syntax by [@openshift-cherrypick-robot](https://github.com/openshift-cherrypick-robot) in [#152](https://github.com/opendatahub-io/MLServer/pull/152)
* Sync release-1.7.x to rhoai-staging by [@github-actions](https://github.com/github-actions)[bot] in [#154](https://github.com/opendatahub-io/MLServer/pull/154)
* chore(deps): update quay.io/aipcc/base-images/cpu docker tag to v3.5.0-ea.1 by [@red-hat-konflux](https://github.com/red-hat-konflux)[bot] in [#153](https://github.com/opendatahub-io/MLServer/pull/153)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#156](https://github.com/opendatahub-io/MLServer/pull/156)
* chore : Manual sync for pyproject.toml files and generate lock files by [@Snomaan6846](https://github.com/Snomaan6846) in [#157](https://github.com/opendatahub-io/MLServer/pull/157)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#158](https://github.com/opendatahub-io/MLServer/pull/158)
* update tag in push job to 3.5-EA1 by [@brettmthompson](https://github.com/brettmthompson) in [#160](https://github.com/opendatahub-io/MLServer/pull/160)
* updating tag regex to allow for -EA* suffix ([#159](https://github.com/opendatahub-io/MLServer/issues/159)) by [@brettmthompson](https://github.com/brettmthompson) in [#162](https://github.com/opendatahub-io/MLServer/pull/162)
* chore(konflux): Bump release tag to odh-v3.5-EA2 by [@github-actions](https://github.com/github-actions)[bot] in [#164](https://github.com/opendatahub-io/MLServer/pull/164)
* Sync release-1.7.x to rhoai-staging by [@github-actions](https://github.com/github-actions)[bot] in [#165](https://github.com/opendatahub-io/MLServer/pull/165)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#166](https://github.com/opendatahub-io/MLServer/pull/166)
* [release-1.7.x] chore(CI): enable all runtime test execution by [@openshift-cherrypick-robot](https://github.com/openshift-cherrypick-robot) in [#167](https://github.com/opendatahub-io/MLServer/pull/167)
* Sync release-1.7.x to rhoai-staging by [@github-actions](https://github.com/github-actions)[bot] in [#168](https://github.com/opendatahub-io/MLServer/pull/168)
* chore : Manual sync for pyproject.toml files and generate lock files by [@Snomaan6846](https://github.com/Snomaan6846) in [#169](https://github.com/opendatahub-io/MLServer/pull/169)
* Sync release-1.7.x to rhoai-staging by [@github-actions](https://github.com/github-actions)[bot] in [#172](https://github.com/opendatahub-io/MLServer/pull/172)

## New Contributors
* [@Jooho](https://github.com/Jooho) made their first contribution in [#17](https://github.com/opendatahub-io/MLServer/pull/17)
* [@github-actions](https://github.com/github-actions)[bot] made their first contribution in [#39](https://github.com/opendatahub-io/MLServer/pull/39)
* [@MohammadiIram](https://github.com/MohammadiIram) made their first contribution in [#66](https://github.com/opendatahub-io/MLServer/pull/66)
* [@RH-steve-grubb](https://github.com/RH-steve-grubb) made their first contribution in [#94](https://github.com/opendatahub-io/MLServer/pull/94)
* [@red-hat-konflux](https://github.com/red-hat-konflux)[bot] made their first contribution in [#153](https://github.com/opendatahub-io/MLServer/pull/153)

**Full Changelog**: https://github.com/opendatahub-io/MLServer/commits/v1.7.1+rhaiv.9

[Changes][v1.7.1+rhaiv.9]


<a id="v1.7.1+rhaiv.8"></a>
## [AIPCC release for MLServer v1.7.1+rhaiv.8](https://github.com/opendatahub-io/MLServer/releases/tag/v1.7.1+rhaiv.8) - 2026-04-08

<!-- Release notes generated using configuration in .github/release.yml at rhoai-staging -->

## What's Changed
* Add support for onnx model format for in Dockerfile by [@Snomaan6846](https://github.com/Snomaan6846) in [#106](https://github.com/opendatahub-io/MLServer/pull/106)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#108](https://github.com/opendatahub-io/MLServer/pull/108)
* update dockerfile.konflux to include labels ([#66](https://github.com/opendatahub-io/MLServer/issues/66)) by [@Snomaan6846](https://github.com/Snomaan6846) in [#109](https://github.com/opendatahub-io/MLServer/pull/109)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#112](https://github.com/opendatahub-io/MLServer/pull/112)
* [rhoai-staging] fix: ONNX model loading failure with KServe Modelcar symlinks by [@openshift-cherrypick-robot](https://github.com/openshift-cherrypick-robot) in [#119](https://github.com/opendatahub-io/MLServer/pull/119)
* Align Dockerfile.konflux improvements and  update aipcc base image for 3.4.0 release by [@Snomaan6846](https://github.com/Snomaan6846) in [#120](https://github.com/opendatahub-io/MLServer/pull/120)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#121](https://github.com/opendatahub-io/MLServer/pull/121)
* Sync release-1.7.x to rhoai-staging by [@github-actions](https://github.com/github-actions)[bot] in [#125](https://github.com/opendatahub-io/MLServer/pull/125)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#128](https://github.com/opendatahub-io/MLServer/pull/128)
* Sync release-1.7.x to rhoai-staging by [@github-actions](https://github.com/github-actions)[bot] in [#132](https://github.com/opendatahub-io/MLServer/pull/132)
* Sync release-1.7.x to rhoai-staging by [@github-actions](https://github.com/github-actions)[bot] in [#136](https://github.com/opendatahub-io/MLServer/pull/136)
* chore: align konflux Dockerfile and bump MLServer/runtimes to 1.7.1+rhaiv.8 by [@Snomaan6846](https://github.com/Snomaan6846) in [#133](https://github.com/opendatahub-io/MLServer/pull/133)


**Full Changelog**: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhai7...v1.7.1+rhaiv.8

[Changes][v1.7.1+rhaiv.8]


<a id="v1.7.1+rhai7"></a>
## [AIPCC release for MLServer v1.7.1+rhai7](https://github.com/opendatahub-io/MLServer/releases/tag/v1.7.1+rhai7) - 2026-03-02

<!-- Release notes generated using configuration in .github/release.yml at rhoai-staging -->

## What's Changed
* ci: Add AIPCC wheels requirements file generation script and github w… by [@Snomaan6846](https://github.com/Snomaan6846) in [#91](https://github.com/opendatahub-io/MLServer/pull/91)
* Regenerate pinned requirements by [@github-actions](https://github.com/github-actions)[bot] in [#92](https://github.com/opendatahub-io/MLServer/pull/92)
* onnx runtime: align version constraints and stabilize test model metadata by [@Snomaan6846](https://github.com/Snomaan6846) in [#99](https://github.com/opendatahub-io/MLServer/pull/99)


**Full Changelog**: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhai6...v1.7.1+rhai7

[Changes][v1.7.1+rhai7]


<a id="v1.7.1+rhai6"></a>
## [AIPCC release for MLServer v1.7.1+rhai6](https://github.com/opendatahub-io/MLServer/releases/tag/v1.7.1+rhai6) - 2026-02-19

<!-- Release notes generated using configuration in .github/release.yml at rhoai-staging -->

## What's Changed
* Cherry pick rhoaieng 46109 by [@Snomaan6846](https://github.com/Snomaan6846) in [#79](https://github.com/opendatahub-io/MLServer/pull/79)
* feat(runtimes): add ONNX runtime support (mlserver_onnx) ([#73](https://github.com/opendatahub-io/MLServer/issues/73)) by [@Snomaan6846](https://github.com/Snomaan6846) in [#85](https://github.com/opendatahub-io/MLServer/pull/85)
* Update MLServer and Runtimes version to 1.7.1+rhai6 by [@Snomaan6846](https://github.com/Snomaan6846) in [#86](https://github.com/opendatahub-io/MLServer/pull/86)


**Full Changelog**: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhai5...v1.7.1+rhai6

[Changes][v1.7.1+rhai6]


<a id="v1.7.1+rhai5"></a>
## [AIPCC release for MLServer v1.7.1+rhai5](https://github.com/opendatahub-io/MLServer/releases/tag/v1.7.1+rhai5) - 2025-12-21

<!-- Release notes generated using configuration in .github/release.yml at rhoai-staging -->

## What's Changed
* Cherry pick chores to staging by [@brettmthompson](https://github.com/brettmthompson) in [#47](https://github.com/opendatahub-io/MLServer/pull/47)
* Update MLServer and Runtimes version to 1.7.1+rhai5 by [@Snomaan6846](https://github.com/Snomaan6846) in [#48](https://github.com/opendatahub-io/MLServer/pull/48)
* Add Dockerfile.konflux for rhoai releases by [@Snomaan6846](https://github.com/Snomaan6846) in [#50](https://github.com/opendatahub-io/MLServer/pull/50)


**Full Changelog**: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhai4...v1.7.1+rhai5

[Changes][v1.7.1+rhai5]


<a id="v1.7.1+rhai4"></a>
## [AIPCC release for MLServer v1.7.1+rhai4](https://github.com/opendatahub-io/MLServer/releases/tag/v1.7.1+rhai4) - 2025-12-17

<!-- Release notes generated using configuration in .github/release.yml at rhoai-staging -->

## What's Changed
* Update MLServer and Runtimes version to 1.7.1+rhai4 by [@Snomaan6846](https://github.com/Snomaan6846) in [#41](https://github.com/opendatahub-io/MLServer/pull/41)


**Full Changelog**: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhai3...v1.7.1+rhai4

[Changes][v1.7.1+rhai4]


<a id="v1.7.1+rhai3"></a>
## [AIPCC release for MLServer v1.7.1+rhai3](https://github.com/opendatahub-io/MLServer/releases/tag/v1.7.1+rhai3) - 2025-12-17



[Changes][v1.7.1+rhai3]


<a id="v1.7.1+rhai2"></a>
## [AIPCC release for MLServer v1.7.1+rhai2](https://github.com/opendatahub-io/MLServer/releases/tag/v1.7.1+rhai2) - 2025-12-15



[Changes][v1.7.1+rhai2]


<a id="v1.7.1+rhai1"></a>
## [AIPCC release for MLServer 1.7.1 (v1.7.1+rhai1)](https://github.com/opendatahub-io/MLServer/releases/tag/v1.7.1+rhai1) - 2025-12-15



[Changes][v1.7.1+rhai1]


[v1.7.1+rhaiv.9]: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhaiv.8...v1.7.1+rhaiv.9
[v1.7.1+rhaiv.8]: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhai7...v1.7.1+rhaiv.8
[v1.7.1+rhai7]: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhai6...v1.7.1+rhai7
[v1.7.1+rhai6]: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhai5...v1.7.1+rhai6
[v1.7.1+rhai5]: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhai4...v1.7.1+rhai5
[v1.7.1+rhai4]: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhai3...v1.7.1+rhai4
[v1.7.1+rhai3]: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhai2...v1.7.1+rhai3
[v1.7.1+rhai2]: https://github.com/opendatahub-io/MLServer/compare/v1.7.1+rhai1...v1.7.1+rhai2
[v1.7.1+rhai1]: https://github.com/opendatahub-io/MLServer/tree/v1.7.1+rhai1

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.9.1 -->
