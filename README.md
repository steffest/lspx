# LSPX

LSPX is a tool that converts an OpenMPT-supported tracker module (mod, s3m, xm, it, mptm) into Amiga LSP (LightSpeed Player) format, while programmatically downsampling all of its samples to a configurable quality target. Composers can leverage the conveniences of modern tracker formats (no sample limit, 48KHz samples, transposing/root note, multi-sample instruments, volume/pitch envelopes, etc.) and use LSPX to convert it to an Amiga-playable format.

## About

Historically, Amiga developers have primarily used ProTracker MOD format for music in games, demos, etc. Although many productions play MOD files directly using libraries like [PTPlayer](https://aminet.net/package/mus/play/ptplayer), there are computationally faster options. [LightSpeed Player](https://github.com/arnaud-carre/LSPlayer) "renders" MOD files into a lower-level format that provides the Amiga audio hardware with pre-baked, frame-by-frame instructions.

Because the LSP format is little more than a set of low-level instructions for instrument offset, length, volume, and pitch, it's feasible to render other tracker module formats into LSP.

## Advantages

Modern tracker formats like MPTM have many advantages over MOD to make composers' lives easier:

- Any number of samples can be used (MOD files have a 32-sample limit).
- More granular finetune.
- Samples can have a root note, meaning notes plotted on the tracker are accurate to the actual tone of the sample.
- Multi-sample instruments: An instrument can assign different samples to be played at different note ranges.
- Instruments can have a volume envelope and pitch envelope, complete with sustain and release behaviors.
- "Global volume" (i.e. master volume) on samples, instruments, and channels, which the volume command mixes with rather than overrides.
- Volume slide and portamento can be triggered via the volume column, freeing the effect column for other uses on those rows.
- Predefined sample cues (offset points) which can be triggered via the volume column.
- The module can use source-quality samples (16-bit, 48000Hz).
- Per-pattern customizable time signature and number of rows in the pattern.
- Other advanced features like macros, MIDI integration, additional effect commands, and more.

LSPX aims to support all of these features. Notably, source-quality samples are automatically downsampled into an Amiga-supported range, individually calculated per sample, based on the lowest and highest pitch played on that sample. That means downsampling is no longer a manual process, nor does the composition of the module have any dependency on the sample rate.

## Restrictions

Modules are still subject to Amiga hardware limitations; only 4 channels are processed, and effects that make no sense on Amiga (e.g. panning and post-processing effects like reverb) are ignored.

## Requirements

LSPX requires [SoX](https://sourceforge.net/projects/sox/) for audio processing. It is available for all major OSes via popular package managers (or, you can download a portable version and manually point LSPX to it using the `--sox-path` parameter).

## Installation

LSPX is available for Python 3.12+ on PyPI:

```console
pip install lspx
```

## Usage

### Command line

```console
lspx [options] input_file
```

Options:

- `-h, --help`: Show help and exit.
- `-q, --quiet`: Run silently (do not output any text to stdout).
- `-o DIRECTORY`: Specify output directory (default: current directory).
- `--no-output`: Process only, do not output any files (dry run).
- `--sample-info`: Print extra sample processing information.
- `--getpos`: Emit LSP GetPos commands at module sequence/order boundaries, allowing the replayer's `LSP_MusicGetPos` function to report the current position.
- `--beat-events`: Emit explicit tracker beat markers for the bundled player's `lsp_get_beat()` function. Beat values wrap every 128 beats. This extension reserves GetPos payloads `128..255`, matching the standard converter's `0..127` sequence-position limit.
- `--version`: Print LSPX version number and exit.
- `--sox-path`: Specify path to SoX executable instead of using system PATH.

Config overrides:

These are config file module options that can also be passed via command line arguments. If the option is also set in the config file, the command line argument will override it.

- `--cfg-ntsc`: Use NTSC clock rate (instead of PAL) when calculating Amiga period values.
- `--cfg-max-rate RATE`: The maximum sample rate at which samples will ever play. LSPX will downsample each sample such that the highest pitch it ever reaches is at or below this sample rate. The default is 28604, which is Amiga period 124 on PAL.
- `--cfg-min-rate RATE`: The target minimum sample rate at which samples will play. LSPX will attempt to downsample each sample such that the lowest pitch it ever reaches in your module is equal to this sample rate. Note that depending on the pitch range of the sample, it may be downsampled further to ensure it doesn't overshoot ``max_rate``.

### Invoking from Python

```py
from lspx.lsp import Module

mod = Module('path/to/module')
bank, score = mod.build() # Final LSP bank (.lsbank) and score (.lsmusic) bytes
```

## Config file

Various processing options can be included in a TOML-formatted config file. Currently, a config file is loaded by sharing the same name and directory as your module, but with the `.toml` extension. For example, if your module is named `cool-song.mptm`, the config file would be named `cool-song.toml` and reside in the same folder as the module.

As of now, config files are primarily useful for manually adjusting the quality of individual samples, e.g. if some samples can afford a reduction in quality in exchange for smaller file size. More options will be supported over time, such as granular trimming controls, [software mixer](https://github.com/DutchRetroGuy/AmigaAudioMixer) support, and post-processing effects like gain adjustment, sidechaining, and arbitrary SoX processing commands.

```toml
# Config options for the module in general
[module]
ntsc = true
max_rate = 28000
min_rate = 22050

# The default options for samples which don't explicitly provide their own
[default]
quality = 1

# Samples are referenced by name
[samples.kick]
quality = 0.5

# Use quotes for sample names with spaces or special characters
[samples."kick 2"]
quality = 0.8
```

## Building from source

Building from source will also build the `openmpt-lspx` submodule, which expects your system to have a standard C/C++ build environment set up, e.g. `make` and `gcc`.

To manually build from source, clone this repository with submodules included, and then `pip install` the local copy:

```console
git clone --recurse-submodule https://github.com/dansalvato/lspx
pip install ./lspx
```

Building on Windows has been tested primarily with Clang, i.e. `CC=clang` and `CXX=clang++`.

## Planned features (not yet supported)

- LSP special commands (set BPM and set position)

## Issues

LSPX has not yet been thoroughly stress-tested, so there will possibly be issues with modules featuring complex behavior, or samples that require a high level of accuracy in their timing. Feel free to report any issues that arise, including a link to the affected module file if applicable.
