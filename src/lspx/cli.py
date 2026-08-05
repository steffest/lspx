from __future__ import annotations
import sys
import shutil
import argparse
import tomllib
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from . import lsp, mpt
from .version import VERSION_FRIENDLY
from .lsp import Module


def main() -> int:
    DESCRIPTION = "Converts an OpenMPT-supported module file (mod, s3m, xm, it, mptm) to Amiga LSP (LightSpeed Player) format."
    CFG_GROUP_DESCRIPTION = "These are config file module options that can also be passed via command line arguments. If the option is also set in the config file, the command line argument will override it."
    HELP_INPUT_FILE = "the input module file"
    HELP_OUTPUT = "specify output directory (default: current directory)"
    HELP_NTSC = "use NTSC clock rate (instead of PAL) when calculating Amiga period values"
    HELP_GETPOS = "emit LSP GetPos commands so playback sequence/order position can be read by the replayer"
    HELP_BEAT_EVENTS = "emit explicit tracker beat markers for the bundled player's lsp_get_beat function"
    HELP_MIN_RATE = "the target minimum sample rate at which samples play their lowest note"
    HELP_MAX_RATE = "the maximum sample rate at which samples will ever play their highest note"
    HELP_QUIET = "run silently (do not output any text to stdout)"
    HELP_NO_OUTPUT = "process only, do not output any files (dry run)"
    HELP_SAMPLE_INFO = "print extra sample processing information"
    HELP_SOX_PATH = "specify path to SoX executable instead of using system PATH"
    HELP_VERSION = "print LSPX version number and exit"


    @dataclass(frozen=True, slots=True)
    class Logger:
        quiet: bool

        def log(self, text: str = '') -> None:
            if not self.quiet:
                print(text)


    @dataclass(slots=True)
    class SampleInfo:
        id: int
        name: str
        orig_size: int # Size in num samples, not bytes
        proc_size: int # Always 8-bit, so samples == bytes anyway
        trim: int
        factor: float
        min_rate: int
        max_rate: int

        def __init__(self, smp: lsp.Sample, msmp: lsp.MSample) -> None:
            self.id = smp.id
            self.name = smp.sample.name
            self.orig_size = smp.sample.length
            self.proc_size = len(msmp.data)
            self.trim = smp.latest_pos
            self.factor = smp.downsample_factor
            rate = mpt.MIX_RATE * smp.downsample_factor
            self.min_rate = round(rate * smp.lowest_pitch)
            self.max_rate = round(rate * smp.highest_pitch)

        def __str__(self) -> str:
            return (f'Sample {self.id}: {self.name}\n'
                    f'Original size: {self.orig_size}\n'
                    f'Processed size: {self.proc_size}\n'
                    f'Trim: {self.trim} ({self.trim / self.orig_size:.2%})\n'
                    f'Downsample factor: {self.factor:.4}\n'
                    f'Min rate: {self.min_rate}\n'
                    f'Max rate: {self.max_rate}')


    def _log_sample_info(mod: Module) -> None:
        for smp in mod.samples.values():
            msmp = mod.msamples[smp.id]
            logger.log(str(SampleInfo(smp, msmp)) + '\n')
        smp_size = sum(sample.sample.length for sample in mod.samples.values())
        msmp_size = sum(len(sample.data) for sample in mod.msamples.values())
        logger.log(f'Total sample savings: {smp_size} -> {msmp_size} ({msmp_size/smp_size:.2%})\n')


    def _found_sox(path: Path | None) -> bool:
        if path is None:
            return shutil.which('sox') is not None
        return shutil.which(path) is not None


    def _error(msg: str) -> None:
        print(f'ERROR: {msg}', file=sys.stderr)


    parser = argparse.ArgumentParser(prog='lspx', description=DESCRIPTION)
    cfg_group = parser.add_argument_group('Config overrides', description=CFG_GROUP_DESCRIPTION)
    cfg_group.add_argument('--cfg-ntsc', action='store_const', const=True, help=HELP_NTSC)
    cfg_group.add_argument('--cfg-min-rate', type=int, help=HELP_MIN_RATE, metavar='RATE')
    cfg_group.add_argument('--cfg-max-rate', type=int, help=HELP_MAX_RATE, metavar='RATE')
    parser.add_argument('--getpos', action='store_const', const=True, default=None, help=HELP_GETPOS)
    parser.add_argument('--beat-events', action='store_const', const=True, default=None, help=HELP_BEAT_EVENTS)
    parser.add_argument('input_file', type=Path, help=HELP_INPUT_FILE)
    parser.add_argument('-q', '--quiet', action='store_true', help=HELP_QUIET)
    parser.add_argument('-o', dest='output_dir', type=Path, metavar='DIRECTORY', default=Path.cwd(), help=HELP_OUTPUT)
    parser.add_argument('--no-output', action='store_true', help=HELP_NO_OUTPUT)
    parser.add_argument('--sample-info', action='store_true', help=HELP_SAMPLE_INFO)
    parser.add_argument('--sox-path', type=Path, help=HELP_SOX_PATH)
    parser.add_argument('--version', action='version', version=VERSION_FRIENDLY, help=HELP_VERSION)
    args = parser.parse_args()
    logger = Logger(args.quiet)
    if not _found_sox(args.sox_path):
        _error("SoX executable not found. Check that sox is in your system PATH, or use --sox-path to manually specify its location.")
        return 1
    if not args.output_dir.is_dir():
        _error(f"Output path is not a directory or does not exist: {args.o}")
        return 1
    logger.log(VERSION_FRIENDLY)
    logger.log(f"Processing {args.input_file.name}...")
    mod_path: Path = args.input_file
    config_path = mod_path.parent/(mod_path.stem + '.toml')
    if not config_path.is_file():
        logger.log("Using default config")
        config = {}
    else:
        logger.log(f"Using config file {config_path}")
        config = tomllib.loads(config_path.read_text())
    cfg_params = {
        'ntsc': args.cfg_ntsc,
        'getpos': args.getpos,
        'beat_events': args.beat_events,
        'min_rate': args.cfg_min_rate,
        'max_rate': args.cfg_max_rate,
    }
    cfg_params = {k: v for k, v in cfg_params.items() if v is not None}
    config['module'] = config.get('module', {}) | cfg_params
    config['module']['sox_path'] = args.sox_path
    lsp = Module(args.input_file, config=config)
    bank, score = lsp.build()
    if not args.quiet and args.sample_info:
        _log_sample_info(lsp)
    input_stem = args.input_file.stem
    if not args.no_output:
        (args.output_dir/f'{input_stem}.lsbank').write_bytes(bank)
        (args.output_dir/f'{input_stem}.lsmusic').write_bytes(score)
    else:
        logger.log("Skipping file output")
    logger.log("Operation finished. Checksum: " + md5(bank + score, usedforsecurity=False).digest()[:4].hex())
    total_size = sum(len(sample.data) for sample in lsp.msamples.values())
    logger.log(f'Total sample size: {total_size} bytes ({round(total_size / 1000)}kb)')
    return 0
