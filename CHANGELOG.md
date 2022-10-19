# Version history
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

# [1.6]
### Added
- Option for indexing ROMS fields by depth instead of index
- Option for indexing ROMS fields by time stamp instead of index
- Option for indexing ROMS fields by longitude and latitude instead of index
### Changed
- Coordinate order of Fields are now (x, y, z, t) instead of (t, z, y, x)


# [1.5]
### Added
- Fields module

## [1.4]
### Added
- Advection module


## [1.3]
### Added
- Four-dimensional coordinate system module 


## [1.2]
### Fixed
- Errors in script `ladim2alcs` that made it impossible to run on dedun 


## [1.1]
### Added
- Script `ladim2alcs` for converting ladim simulations to finite time lyapunov
  exponents


## [1.0] - 2022-08-15
### Added
- Basic project files
