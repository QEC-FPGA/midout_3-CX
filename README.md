# midout_3-CX

Extensions to midout for three-coupler surface-code circuits and lattice-surgery operations.

This repository contains the midout implementation used by the companion paper
*Universal logical operations and their circuit implementations for surface codes
on trivalent hardware architectures* (三连通硬件架构上表面码的通用逻辑操作及其线路实现).

The [paper repository](https://github.com/QEC-FPGA/trivalent-logical-operations)
provides experiment drivers, data, frozen circuits, figures and reproducibility
instructions. It pins this repository as a Git submodule.

## Source and version

This is a fork of [Strilanc/midout](https://github.com/Strilanc/midout), starting
from upstream commit `941a0ca5b9fd2c0a122eef2a8f41d0d3188ed265`.
The upstream commit and attribution remain in the Git history. The original
[project description](https://github.com/Strilanc/midout/blob/941a0ca5b9fd2c0a122eef2a8f41d0d3188ed265/README.md)
is available at that baseline. This branch contains only the 22 source files
used by the paper, identified byte-for-byte in `research-snapshot.json`.
Other upstream circuit families, benchmark data and command-line tools are
outside this dependency snapshot.

The research extensions provide joint-measurement schedules, detector selection
and ordering, and circuit-generation helpers used by the paper. The Python
package remains `midout` under `src/`.

## Environment

The companion experiment environment uses Python 3.12:

```bash
python -m pip install -r requirements-research.txt
PYTHONPATH=src python -c "from midout.planar import _three_coupler_surface_codes"
```

Paper experiments use the companion repository's pinned environment and
frozen inputs. To reproduce the original midout paper, use the upstream
repository and its original instructions.

## Attribution and license scope

The upstream project accompanies *Relaxing Hardware Requirements for Surface
Code Circuits using Time-dynamics* by Matt McEwen, Dave Bacon and Craig Gidney.
Preserve its attribution and cite it when using the relevant methods.

[LICENSE.md](LICENSE.md) grants MIT only for original contributions in which the
research authors hold the necessary rights. It does not relicense upstream code.
No standalone open-source license is present in the identified upstream commit;
GitHub's platform-granted viewing and fork rights remain unaffected.
