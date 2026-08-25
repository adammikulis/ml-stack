# Changelog

## [0.1.6](https://github.com/adammikulis/ml-stack/compare/v0.1.5...v0.1.6) (2026-08-25)


### Features

* publish the packages to PyPI ([69c1020](https://github.com/adammikulis/ml-stack/commit/69c102029893be2c4e09c604c65724fae897c8f8))


### Bug Fixes

* pass the PyPI secret to the workflow that uploads ([abd8912](https://github.com/adammikulis/ml-stack/commit/abd8912344504ab65ddbeea898141832cfc941f3))

## [0.1.5](https://github.com/adammikulis/ml-stack/compare/v0.1.4...v0.1.5) (2026-08-25)


### Features

* Every step of setting up has a back button, and going back keeps what you already answered ([acbc3e8](https://github.com/adammikulis/ml-stack/commit/acbc3e8))
* One page lists the clusters this machine is in, joins another with a passphrase, and leaves one ([abd1ad4](https://github.com/adammikulis/ml-stack/commit/abd1ad4))
* A machine can belong to no cluster at all: it runs models and trains on its own, and answers to that machine only ([33eb607](https://github.com/adammikulis/ml-stack/commit/33eb607))
* What the machine does and when it starts are asked on separate pages ([33da8f6](https://github.com/adammikulis/ml-stack/commit/33da8f6))
* What it can run and what it does unattended are asked on separate pages, and the training list says how much it is about to download ([69ee59b](https://github.com/adammikulis/ml-stack/commit/69ee59b))
* A machine takes one job at a time ([a6400a1](https://github.com/adammikulis/ml-stack/commit/a6400a1))
* A passphrase can be five characters ([eed12dc](https://github.com/adammikulis/ml-stack/commit/eed12dc))
* The small model that guesses ahead is copied from a machine on your network, as the model itself is ([65053fa](https://github.com/adammikulis/ml-stack/commit/65053fa))
* Each download is named after the release it came from ([03aa293](https://github.com/adammikulis/ml-stack/commit/03aa293))
* Releases carry their downloads under the machine they are for ([52204a7](https://github.com/adammikulis/ml-stack/commit/52204a7))


### Bug Fixes

* The close button asks whether to keep running or quit, instead of leaving the window unresponsive ([7790d9a](https://github.com/adammikulis/ml-stack/commit/7790d9a))
* A machine set up before this release keeps the cluster it joined ([006d40d](https://github.com/adammikulis/ml-stack/commit/006d40d))
* The same passphrase makes the same cluster whether it was typed during setup or on the cluster screen ([9a13dd7](https://github.com/adammikulis/ml-stack/commit/9a13dd7))
* Changing when the machine starts replaces what the last answer installed, rather than leaving both ([9967ce1](https://github.com/adammikulis/ml-stack/commit/9967ce1))
* Starting at login is refused, with the reason, on a machine where the command it would install cannot run ([c315783](https://github.com/adammikulis/ml-stack/commit/c315783))
