<div align="center">

<img src="static\logo.png#gh-light-mode-only" width="400">
<img src="static\logo_dark.png#gh-dark-mode-only" width="400">

<h2>Break the Rules to Beat the Benchmark</h2>

</div>

<div align="center">
<img src="static\demo.gif" width="300">
</div>
<div align="center">
 <a href="https://arxiv.org/pdf/2407.13729"style="margin-right:20px;">Paper</a> &nbsp;&nbsp;&nbsp;
 <a href="https://x.com/nacloos/status/1814709193221304403">Tweet</a> 
</div>


## Installation
```
git clone https://github.com/nacloos/baba-is-ai
cd baba-is-ai
pip install -e .
```

## Usage
To play an environment, run this command:
```bash
python baba/play.py --env two_room-break_stop-make_win
```
The `--env` argument specifies the ID of the environment. Once the game opens, use the arrow keys to move the agent.

You can also create a Gym environment object:
```python
import baba

env_id = "two_room-break_stop-make_win"
env = baba.make(f"env/{env_id}")
```

To list all available environment IDs, run this code:
```python
import baba

print(baba.make("env/*").keys())
```

## Seeded rule-composition extension

This fork adds ten procedurally instantiated task families. They combine the
engine's existing `YOU`, `WIN`, `STOP`, color-prefix, and pushable-word
semantics; no new rule or movement mechanic was added.

| Environment ID | Size | Required reasoning |
|---|---:|---|
| `env/composition-01-target-lock` | 8x8 | Remove `STOP` while preserving `WIN` |
| `env/composition-02-stop-to-win` | 8x8 | Replace `STOP` with a loose `WIN` word |
| `env/composition-03-reuse-is` | 13x9 | Reuse `IS` from the wall rule |
| `env/composition-04-reuse-win` | 8x8 | Move `WIN` from a decoy rule |
| `env/composition-05-cross-rule` | 13x9 | Break one of two crossing rules |
| `env/composition-06-color-scope` | 8x8 | Narrow `STOP` with a color prefix |
| `env/composition-07-one-token-fork` | 8x8 | Commit one `WIN` word to the useful branch |
| `env/composition-08-control-handoff` | 8x8 | Transfer `YOU` to one colored object |
| `env/composition-09-ordered-assembly` | 8x8 | Assemble words in a forced order |
| `env/composition-10-control-relay` | 13x9 | Transfer control, restore Baba, then finish |

Every family has four validated geometry templates. A seed chooses a template
and samples object identities, colors, and distractors reproducibly:

```python
import baba

env = baba.make("env/composition-08-control-handoff")
env.reset(seed=123)
image = env.render(mode="rgb_array")
print(env.generation_params)
```

`reference_solution` and `generation_params` are validation metadata. They are
not included in BALROG's agent observations.

### Validation

Run the normal suite from this repository in the Python environment where the
package is installed:

```bash
python -m unittest discover -s tests
```

The normal suite replays generated solutions, checks reproducibility and
diversity, exercises color/control transitions, and proves shortest paths for
the smaller templates. Exhaustive checks for all larger templates are opt-in:

```bash
BABA_RUN_SLOW_TESTS=1 python -m unittest discover -s tests
```

The semantic shortest-solution analyzer can also be run directly:

```bash
python -m baba.solution_analysis --workers 4
```

## Citation
If you use this project in your research, please cite:
```
@inproceedings{
  cloos2024baba,
  title={Baba Is AI: Break the Rules to Beat the Benchmark},
  author={Nathan Cloos and Meagan Jens and Michelangelo Naim and Yen-Ling Kuo and Ignacio Cases and Andrei Barbu and Christopher J Cueva},
  booktitle={ICML 2024 Workshop on LLMs and Cognition},
  year={2024}
}
```
